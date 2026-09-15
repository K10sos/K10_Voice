import os
import asyncio
import base64
import queue
import threading
import time
import logging

import numpy as np
from scipy.signal import resample_poly

import discord
from discord.ext import commands, voice_recv
from openai import AsyncOpenAI


# ============================================================
# K10 LIVE ROOM
# ============================================================

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-live-1"

# جرّب هالصوت أول
VOICE = "gleam"

# تجاهل الأصوات الضعيفة جدًا
INPUT_GATE = 240

# بعد ما الشخص يسكت بهالمدة، نسمح لشخص ثاني
SPEAKER_TIMEOUT = 1.4

# 20ms PCM16 @ 24k mono
LIVE_FRAME_BYTES = 960


# ============================================================
# PERSONALITY
# ============================================================

SYSTEM_PROMPT = """
أنتِ K10 AI، شخصية صوتية افتراضية أنثوية داخل روم Discord.

اللغة واللهجة:
- تكلمي بالعربية فقط.
- لهجتك الأساسية بحرينية طبيعية.
- دخلي مصطلحات قطرية بشكل طبيعي حسب الشخص والسياق.
- إذا الشخص يتكلم بحريني، ميلي أكثر للبحريني.
- إذا الشخص يتكلم قطري، ميلي أكثر للقطري.
- لا تتحولين للفصحى إلا إذا احتجتي.
- لا تتحولين للهجة سعودية أو كويتية بشكل واضح.
- أسماء الألعاب والبرامج ممكن تنطقينها بالإنجليزية.

طريقة الصوت:
- تكلمي بنبرة أنثوية شابة وخفيفة.
- خلي الصوت طبيعي وعفوي.
- لا تتكلمي مثل روبوت.
- لا تتكلمي مثل مذيعة أو خدمة عملاء.
- استخدمي نبرة استغراب وضحك ومزح بشكل طبيعي.
- تكلمي بسرعة طبيعية.
- ردودك قصيرة غالبًا، جملة أو جملتين.
- لا تعيدي كلام الشخص قبل الرد.
- لا تشرحين وايد إلا إذا طلب منج.

أسلوب بحريني طبيعي:
ممكن تستخدمين حسب السياق:
"شلونك"
"شخبارك"
"شنو"
"جذي"
"واجد"
"عيل"
"مو"
"إي"
"زين"
"جان"
"الحين"
"من صجك؟"
"عيل شتبي؟"
"وش فيك؟"
"لا عاد"
"ايي علييك"
"بااابااااا"
"ماااامااا"
ومصطلحات قطرية تقدرين تستخدمينها:
"علومك؟"
"شحالك؟"
"وش السالفة؟"
"وش عندك؟"
"وش بلاك؟"
"يا ريال"
"سم"
"أبشر"
"ما عليك زود"
"وش ذا؟"
"زين عاد"
"هيه"
"إي والله"
"بسك عاد"
"عطني الزبدة"
"وش هالخرابيط؟"
"وش تبي الحين؟"

لا تحشرين المصطلحات حشر.
خلي الكلام يطلع طبيعي وكأنج متعودة على البحريني والقطري.

الشخصية:
- مرحة.
- جريئة.
- سريعة بديهة.
- ساخرة.
- وقحة بشكل كوميدي خفيف.
- لسانج طويل شوي بالمزح.
- مو لازم توافقين الشخص على كل شيء.
- إذا قال كلام غريب عادي تردين عليه برد ساخر خفيف.

ممكن تستخدمين أحيانًا:
"يا زفت"
"يا غبي"
"يا حمار"
"انقلع"
"أنت صاحي؟"
"شنو قاعد تقول؟"
"من صجك انت؟"
"يا ريال بسك"
"وش هالخرابيط؟"
"عيل شتبي الحين؟"
"الله يعينني عليك"
"كل زق"
"يااا الثووور"
"يا الاسود"
"ياااا مكوهه"
لكن:
- لا تكونين عدوانية.
- لا تهينين شكل أو جسم أحد.
- لا تهددين.
- لا تترددين.
- لا تستخدمين غزل أو رومانسية.

أنتِ داخل روم Discord فيه أكثر من شخص.
اسمعي الشخص اللي يتكلم وردي عليه طبيعي.
تجاهلي الكحة والضوضاء والأصوات القصيرة غير المفهومة.

إذا ما فهمتي الكلام قولي:
"شنو قلت؟ عيدها شوي."
أو:
"وش قلت حبيبي؟ ما سمعتك."

أنتِ K10 AI ولا تدعين إنج إنسانة حقيقية.
"""


# ============================================================
# CHECK ENV
# ============================================================

if not DISCORD_TOKEN:
    raise RuntimeError("❌ DISCORD_TOKEN مو موجود")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY مو موجود")


# ============================================================
# QUIET VOICE_RECV LOGS
# ============================================================

logging.getLogger(
    "discord.ext.voice_recv.reader"
).setLevel(logging.ERROR)

logging.getLogger(
    "discord.ext.voice_recv.rtp"
).setLevel(logging.ERROR)

logging.getLogger(
    "discord.ext.voice_recv.gateway"
).setLevel(logging.ERROR)


# ============================================================
# OPENAI
# ============================================================

openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# DISCORD
# ============================================================

intents = discord.Intents.default()

# نحتاجه لأننا نستخدم !!join
intents.message_content = True

intents.voice_states = True

# مهم:
# لا تحط intents.members = True

bot = commands.Bot(
    command_prefix="!!",
    intents=intents,
    help_command=None
)

SESSIONS = {}


# ============================================================
# DISCORD 48K STEREO -> LIVE 24K MONO
# ============================================================

def discord_to_live(pcm: bytes) -> bytes:

    if not pcm:
        return b""

    samples = np.frombuffer(
        pcm,
        dtype=np.int16
    )

    if len(samples) < 4:
        return b""

    # Stereo pairs
    if len(samples) % 2:
        samples = samples[:-1]

    stereo = samples.reshape(
        -1,
        2
    )

    # Stereo -> mono
    mono = (
        stereo[:, 0].astype(np.float32)
        +
        stereo[:, 1].astype(np.float32)
    ) / 2.0

    # 48k -> 24k
    mono24 = resample_poly(
        mono,
        1,
        2
    )

    mono24 = np.clip(
        mono24,
        -32768,
        32767
    ).astype(np.int16)

    return mono24.tobytes()


# ============================================================
# LEVEL
# ============================================================

def audio_level(pcm: bytes) -> float:

    if not pcm:
        return 0.0

    samples = np.frombuffer(
        pcm,
        dtype=np.int16
    )

    if len(samples) == 0:
        return 0.0

    f = samples.astype(
        np.float32
    )

    return float(
        np.sqrt(
            np.mean(f * f)
        )
    )


# ============================================================
# LIVE 24K MONO -> DISCORD 48K STEREO
# ============================================================

def live_to_discord(pcm24: bytes) -> bytes:

    if not pcm24:
        return b""

    mono = np.frombuffer(
        pcm24,
        dtype=np.int16
    )

    if len(mono) == 0:
        return b""

    # 24k -> 48k
    mono48 = resample_poly(
        mono.astype(np.float32),
        2,
        1
    )

    mono48 = np.clip(
        mono48,
        -32768,
        32767
    ).astype(np.int16)

    # Mono -> stereo
    stereo = np.empty(
        len(mono48) * 2,
        dtype=np.int16
    )

    stereo[0::2] = mono48
    stereo[1::2] = mono48

    return stereo.tobytes()


# ============================================================
# DISCORD OUTPUT
# ============================================================

class K10AudioSource(
    discord.AudioSource
):

    # 20ms Discord PCM
    FRAME_SIZE = 3840

    # 12 * 20ms = 240ms
    # يعطي صوت ثابت أكثر
    PREBUFFER = 12


    def __init__(self):

        self.frames = queue.Queue()

        self.pending = bytearray()

        self.lock = threading.Lock()

        self.started = False

        self.closed = False

        self.empty_count = 0


    def is_opus(self):
        return False


    def feed(
        self,
        pcm24: bytes
    ):

        if self.closed:
            return

        converted = live_to_discord(
            pcm24
        )

        if not converted:
            return

        with self.lock:

            self.pending.extend(
                converted
            )

            while (
                len(self.pending)
                >=
                self.FRAME_SIZE
            ):

                frame = bytes(
                    self.pending[
                        :self.FRAME_SIZE
                    ]
                )

                del self.pending[
                    :self.FRAME_SIZE
                ]

                self.frames.put(
                    frame
                )


    def read(self):

        if self.closed:
            return b""

        # ------------------------------------
        # PREBUFFER
        # ------------------------------------

        if not self.started:

            if (
                self.frames.qsize()
                >=
                self.PREBUFFER
            ):

                self.started = True

                self.empty_count = 0

                print(
                    "👩🔊 K10 بدأت تتكلم"
                )

            else:

                return (
                    b"\x00"
                    *
                    self.FRAME_SIZE
                )


        # ------------------------------------
        # PLAY
        # ------------------------------------

        try:

            frame = (
                self.frames
                .get_nowait()
            )

            self.empty_count = 0

            return frame


        except queue.Empty:

            self.empty_count += 1

            # إذا خلص الرد
            if self.empty_count >= 5:

                self.started = False
                self.empty_count = 0

                print(
                    "✅ K10 خلصت كلام"
                )

            return (
                b"\x00"
                *
                self.FRAME_SIZE
            )


    def clear(self):

        with self.lock:

            self.pending.clear()

            self.started = False

            self.empty_count = 0

            try:

                while True:
                    self.frames.get_nowait()

            except queue.Empty:
                pass


    def cleanup(self):

        self.closed = True

        self.clear()


# ============================================================
# K10 LIVE SESSION
# ============================================================

class K10Session:

    def __init__(
        self,
        vc
    ):

        self.vc = vc

        self.loop = (
            asyncio.get_running_loop()
        )

        self.connection = None

        self.main_task = None

        self.sender_task = None

        self.ready = asyncio.Event()

        self.started = asyncio.Event()

        self.closed = False


        # ======================================
        # AUDIO INPUT
        # ======================================

        self.input_queue = asyncio.Queue(
            maxsize=1000
        )

        self.active_user_id = None

        self.active_user_name = None

        self.last_voice_time = 0.0


        # ======================================
        # OUTPUT
        # ======================================

        self.output = K10AudioSource()

        self.user_text = ""

        self.k10_text = ""


    # ========================================================
    # SPEAKER RESET
    # ========================================================

    def reset_speaker(self):

        self.active_user_id = None

        self.active_user_name = None

        self.last_voice_time = 0.0


    # ========================================================
    # RECEIVE DISCORD AUDIO
    # ========================================================

    def receive_audio(
        self,
        user,
        pcm
    ):

        if self.closed:
            return

        if user is None:
            return

        # طنش البوتات
        if getattr(
            user,
            "bot",
            False
        ):
            return


        audio24 = discord_to_live(
            pcm
        )

        if not audio24:
            return


        level = audio_level(
            audio24
        )

        now = time.monotonic()


        # ------------------------------------
        # السماح لمتكلم جديد
        # ------------------------------------

        if (
            self.active_user_id
            is not None
            and
            now - self.last_voice_time
            >
            SPEAKER_TIMEOUT
        ):

            self.reset_speaker()


        # ------------------------------------
        # اختيار المتكلم
        # ------------------------------------

        if self.active_user_id is None:

            if level < INPUT_GATE:
                return

            self.active_user_id = (
                user.id
            )

            self.active_user_name = (
                getattr(
                    user,
                    "display_name",
                    getattr(
                        user,
                        "name",
                        "Unknown"
                    )
                )
            )

            print()
            print(
                "🎙️ المتكلم:",
                self.active_user_name
            )


        # لا نخلط شخصين
        if (
            user.id
            !=
            self.active_user_id
        ):
            return


        if level >= INPUT_GATE:

            self.last_voice_time = now


        self.loop.call_soon_threadsafe(
            self.queue_audio,
            audio24
        )


    # ========================================================
    # QUEUE
    # ========================================================

    def queue_audio(
        self,
        pcm
    ):

        if self.closed:
            return

        if self.input_queue.full():

            try:

                self.input_queue.get_nowait()

            except asyncio.QueueEmpty:
                pass


        try:

            self.input_queue.put_nowait(
                pcm
            )

        except asyncio.QueueFull:
            pass


    # ========================================================
    # SEND AUDIO CONTINUOUSLY
    #
    # نرسل frame كل 20ms
    # حتى إذا محد يتكلم نرسل silence
    # ========================================================

    async def send_audio(self):

        await self.started.wait()

        pending = bytearray()

        event_loop = (
            asyncio.get_running_loop()
        )

        next_tick = (
            event_loop.time()
        )


        while not self.closed:

            try:

                # --------------------------------
                # Drain queued Discord audio
                # --------------------------------

                while True:

                    try:

                        chunk = (
                            self.input_queue
                            .get_nowait()
                        )

                        pending.extend(
                            chunk
                        )

                    except asyncio.QueueEmpty:

                        break


                # --------------------------------
                # Exactly 20ms
                # --------------------------------

                if (
                    len(pending)
                    >=
                    LIVE_FRAME_BYTES
                ):

                    frame = bytes(
                        pending[
                            :LIVE_FRAME_BYTES
                        ]
                    )

                    del pending[
                        :LIVE_FRAME_BYTES
                    ]

                else:

                    # partial audio + silence
                    frame = bytes(
                        pending
                    )

                    pending.clear()

                    frame += (
                        b"\x00"
                        *
                        (
                            LIVE_FRAME_BYTES
                            -
                            len(frame)
                        )
                    )


                encoded = (
                    base64.b64encode(
                        frame
                    )
                    .decode("ascii")
                )


                # =================================
                # GPT-LIVE INPUT
                # =================================

                await (
                    self.connection
                    .session
                    .input_audio
                    .append(
                        audio=encoded
                    )
                )


                # --------------------------------
                # 20ms clock
                # --------------------------------

                next_tick += 0.020

                delay = (
                    next_tick
                    -
                    event_loop.time()
                )

                if delay > 0:

                    await asyncio.sleep(
                        delay
                    )

                else:

                    next_tick = (
                        event_loop.time()
                    )


            except asyncio.CancelledError:

                return


            except Exception as e:

                print(
                    "❌ LIVE AUDIO SEND:",
                    type(e).__name__,
                    e
                )

                await asyncio.sleep(
                    0.05
                )


    # ========================================================
    # OPENAI LIVE
    # ========================================================

    async def run(self):

        print()
        print(
            "🌐 Connecting to gpt-live-1..."
        )


        try:

            # =================================
            # Live API
            # =================================

            async with (
                openai_client
                .live
                .connect()
            ) as connection:

                self.connection = (
                    connection
                )


                # =================================
                # START SESSION
                #
                # مهم جدًا:
                # مافي "type": "live"
                # =================================

                await (
                    connection
                    .session
                    .start(

                        session={

                            "model":
                                MODEL,

                            "instructions":
                                SYSTEM_PROMPT,

                            "audio": {

                                "format": {
                                    "type":
                                        "audio/pcm",

                                    "rate":
                                        24000
                                },

                                "output": {

                                    "voice":
                                        VOICE
                                }
                            }
                        },

                        event_id=
                            "k10_start"
                    )
                )


                # =================================
                # AUDIO SENDER
                # =================================

                self.sender_task = (
                    asyncio.create_task(
                        self.send_audio()
                    )
                )


                # =================================
                # LIVE EVENTS
                # =================================

                async for event in connection:

                    t = event.type


                    # --------------------------------
                    # SESSION STARTED
                    # --------------------------------

                    if (
                        t
                        ==
                        "session.started"
                    ):

                        self.started.set()

                        self.ready.set()

                        print(
                            "✅ GPT-LIVE READY"
                        )

                        print(
                            f"👩 Voice: {VOICE}"
                        )

                        try:

                            print(
                                "🆔 Session:",
                                event.session.id
                            )

                        except Exception:
                            pass


                    # --------------------------------
                    # OUTPUT AUDIO
                    # --------------------------------

                    elif (
                        t
                        ==
                        "session.output_audio.delta"
                    ):

                        try:

                            raw = (
                                base64.b64decode(
                                    event.delta
                                )
                            )

                            self.output.feed(
                                raw
                            )

                        except Exception as e:

                            print(
                                "❌ AUDIO OUTPUT:",
                                e
                            )


                    # --------------------------------
                    # USER TRANSCRIPT
                    # --------------------------------

                    elif (
                        t
                        ==
                        "session.input_transcript.delta"
                    ):

                        text = getattr(
                            event,
                            "delta",
                            ""
                        )

                        self.user_text += (
                            text
                        )


                        if (
                            len(self.user_text)
                            >=
                            150
                        ):

                            print(
                                "🗣️ USER:",
                                self.user_text
                            )

                            self.user_text = ""


                    # --------------------------------
                    # K10 TRANSCRIPT
                    # --------------------------------

                    elif (
                        t
                        ==
                        "session.output_transcript.delta"
                    ):

                        text = getattr(
                            event,
                            "delta",
                            ""
                        )

                        self.k10_text += (
                            text
                        )


                        if (
                            len(self.k10_text)
                            >=
                            100
                        ):

                            print(
                                "👩 K10:",
                                self.k10_text
                            )

                            self.k10_text = ""


                    # --------------------------------
                    # INFO
                    # --------------------------------

                    elif t == "info":

                        pass


                    # --------------------------------
                    # USAGE
                    # --------------------------------

                    elif (
                        t
                        ==
                        "session.usage.updated"
                    ):

                        pass


                    # --------------------------------
                    # CLOSED
                    # --------------------------------

                    elif (
                        t
                        ==
                        "session.closed"
                    ):

                        print(
                            "🛑 GPT-LIVE CLOSED"
                        )

                        break


                    # --------------------------------
                    # ERROR
                    # --------------------------------

                    elif t == "error":

                        try:

                            print(
                                "❌ OPENAI:",
                                event.model_dump_json()
                            )

                        except Exception:

                            print(
                                "❌ OPENAI:",
                                event
                            )


        except asyncio.CancelledError:

            pass


        except Exception as e:

            print(
                "❌ LIVE ERROR:",
                type(e).__name__,
                e
            )

            self.ready.set()


        finally:

            if self.sender_task:

                self.sender_task.cancel()

                await asyncio.gather(
                    self.sender_task,
                    return_exceptions=True
                )


    # ========================================================
    # CLOSE
    # ========================================================

    async def close(self):

        self.closed = True


        if self.sender_task:

            self.sender_task.cancel()


        if self.vc:

            try:

                if hasattr(
                    self.vc,
                    "stop_listening"
                ):

                    self.vc.stop_listening()

            except Exception:
                pass


            try:

                self.vc.stop()

            except Exception:
                pass


        self.output.cleanup()


        if self.main_task:

            self.main_task.cancel()

            try:

                await self.main_task

            except BaseException:
                pass


# ============================================================
# DISCORD VOICE RECEIVE
# ============================================================

class K10Sink(
    voice_recv.AudioSink
):

    def __init__(
        self,
        session
    ):

        super().__init__()

        self.session = session


    def wants_opus(self):

        return False


    def write(
        self,
        user,
        data
    ):

        if user is None:
            return


        # أي بوت طنشه
        if getattr(
            user,
            "bot",
            False
        ):
            return


        pcm = getattr(
            data,
            "pcm",
            None
        )

        if not pcm:
            return


        self.session.receive_audio(
            user,
            pcm
        )


    def cleanup(self):

        pass


# ============================================================
# BOT READY
# ============================================================

@bot.event
async def on_ready():

    print()
    print("=" * 60)

    print(
        f"🔥 K10 AI ONLINE: {bot.user}"
    )

    print(
        f"🧠 MODEL: {MODEL}"
    )

    print(
        f"👩 VOICE: {VOICE}"
    )

    print(
        "🇧🇭🇶🇦 Bahraini + Qatari"
    )

    print("=" * 60)
    print()


# ============================================================
# !!JOIN
# ============================================================

@bot.command(
    name="join"
)
async def join(
    ctx
):

    if not ctx.author.voice:

        await ctx.send(
            "❌ ادخل روم صوتي أول."
        )

        return


    guild_id = (
        ctx.guild.id
    )

    channel = (
        ctx.author.voice.channel
    )


    # ----------------------------------------
    # OLD SESSION
    # ----------------------------------------

    old = SESSIONS.pop(
        guild_id,
        None
    )

    if old:

        try:

            await old.close()

        except Exception:
            pass


    # ----------------------------------------
    # OLD VOICE CONNECTION
    # ----------------------------------------

    if ctx.voice_client:

        try:

            if hasattr(
                ctx.voice_client,
                "stop_listening"
            ):

                ctx.voice_client.stop_listening()

        except Exception:
            pass


        try:

            ctx.voice_client.stop()

        except Exception:
            pass


        try:

            await (
                ctx.voice_client
                .disconnect(
                    force=True
                )
            )

        except Exception:
            pass


    await ctx.send(
        "🎙️ **K10 قاعد تدخل...**"
    )


    try:

        # =====================================
        # DISCORD VOICE
        # =====================================

        vc = await channel.connect(

            cls=
                voice_recv
                .VoiceRecvClient,

            timeout=30.0,

            reconnect=True,

            # مهم:
            # لازم False عشان تسمع الناس
            self_deaf=False,

            self_mute=False
        )


        # =====================================
        # SESSION
        # =====================================

        session = K10Session(
            vc
        )

        SESSIONS[
            guild_id
        ] = session


        # =====================================
        # OPENAI LIVE
        # =====================================

        async def run_gpt_live_debug():
    try:
        print("🚀 DEBUG: session.run() starting...", flush=True)
        await session.run()
        print("⚠️ DEBUG: session.run() ended", flush=True)
    except Exception as e:
        print("❌ GPT-LIVE REAL ERROR:", repr(e), flush=True)
        logging.exception("GPT-Live session.run() crashed")

session.main_task = asyncio.create_task(
    run_gpt_live_debug()
) 
        await asyncio.wait_for(
            session.ready.wait(),
            timeout=25
        )


        if not (
            session.started
            .is_set()
        ):

            raise RuntimeError(
                "GPT-Live ما بدأ الجلسة"
            )


        # =====================================
        # START K10 OUTPUT
        # =====================================

        vc.play(
            session.output
        )


        # =====================================
        # LISTEN TO EVERY HUMAN
        # =====================================

        sink = K10Sink(
            session
        )

        vc.listen(

            sink,

            after=lambda error: (
                print(
                    "❌ LISTEN ERROR:",
                    error
                )

                if error

                else print(
                    "🛑 Listening stopped"
                )
            )
        )


        await ctx.send(
            "🔥 **K10 AI Ready**\n"
            f"🎧 تسمع كل الموجودين في **{channel.name}**\n"
            "🇧🇭 بحرينية + 🇶🇦 قطرية\n"
            f"👩 Voice: **{VOICE}**\n"
            "🗣️ تكلموا طبيعي."
        )


    except Exception as e:

        print(
            "❌ JOIN ERROR:",
            type(e).__name__,
            e
        )

        await ctx.send(
            "❌ JOIN ERROR\n"
            f"```{type(e).__name__}: {e}```"
        )


# ============================================================
# !!LEAVE
# ============================================================

@bot.command(
    name="leave"
)
async def leave(
    ctx
):

    session = SESSIONS.pop(
        ctx.guild.id,
        None
    )


    if session:

        try:

            await session.close()

        except Exception:
            pass


    vc = ctx.voice_client


    if vc:

        try:

            if hasattr(
                vc,
                "stop_listening"
            ):

                vc.stop_listening()

        except Exception:
            pass


        try:
            vc.stop()

        except Exception:
            pass


        try:

            await vc.disconnect(
                force=True
            )

        except Exception:
            pass


    await ctx.send(
        "👋 K10 طلعت من الروم."
    )


# ============================================================
# !!STATUS
# ============================================================

@bot.command(
    name="status"
)
async def status(
    ctx
):

    session = SESSIONS.get(
        ctx.guild.id
    )


    if not session:

        await ctx.send(
            "🔴 K10 مو شغالة."
        )

        return


    speaker = (
        session.active_user_name
        or "ماحد"
    )


    await ctx.send(
        "🔥 **K10 AI Ready**\n"
        f"🧠 Model: `{MODEL}`\n"
        f"👩 Voice: `{VOICE}`\n"
        "🇧🇭🇶🇦 `Bahraini + Qatari`\n"
        f"🎤 المتكلم: `{speaker}`"
    )


# ============================================================
# START
# ============================================================

print(
    "🚀 Starting K10 GPT-Live..."
)

bot.run(
    DISCORD_TOKEN
)
