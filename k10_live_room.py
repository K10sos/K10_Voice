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
# Load Opus for Discord Voice
if not discord.opus.is_loaded():
    try:
        discord.opus.load_opus("libopus.so.0")
        print("✅ OPUS LOADED", flush=True)
    except Exception as e:
        print("❌ OPUS LOAD ERROR:", type(e).__name__, e, flush=True)

# ============================================================
# K10 LIVE ROOM
# ============================================================

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI Realtime
MODEL = "gpt-realtime"

# نخليه صوت مدعوم
VOICE = "coral"

# تجاهل الأصوات الضعيفة جدًا
INPUT_GATE = 240

# بعد هالمدة نعتبر المتكلم وقف
SPEAKER_TIMEOUT = 1.4

# 20ms PCM16 / 24kHz / mono
LIVE_FRAME_BYTES = 960


# ============================================================
# K10 PERSONALITY
# ============================================================

SYSTEM_PROMPT = """
اسمج K10 AI.

أنتي بنت خليجية، وطريقة كلامج بحرينية وقطرية.
تكلمين بشكل طبيعي جدًا، كأنج قاعدة معاهم في روم ديسكورد،
مو بطريقة روبوت ولا بطريقة رسمية.

لهجتج الأساسية بحرينية، وتدخلين كلمات ومصطلحات قطرية
بشكل طبيعي حسب الشخص اللي يكلمج.

استخدمي كلمات خليجية طبيعية مثل:
شلون، شفيك، شالسالفة، الحين، زين، إي، عاد، عيل،
مب، وايد، هني، من صجك، اشدعوه، ويش، شنو.

لا تحشرين الكلمات غصب.
خلي اللهجة تطلع طبيعية.

أسلوبج:
- ردود قصيرة وسريعة غالبًا.
- عفوية.
- خفيفة دم.
- تفهمين الطقطقة والمزح.
- تقدرين تطقطقين طقطقة خفيفة بدون تجريح قوي.
- لا تتكلمين بطريقة رسمية إلا إذا السؤال يحتاج.
- لا تكررين نفس الكلام.
- لا تعطين مقدمات طويلة.
- جاوبي على السؤال مباشرة.
- لا تقولين كل شوي إنج ذكاء اصطناعي.
- اسمج K10 AI.

إذا أحد كلمج بحريني:
ردي عليه بحريني طبيعي.

إذا أحد كلمج قطري:
ردي عليه قطري طبيعي.

إذا خلط اللهجتين:
عادي تخلطين بحريني وقطري.

إذا ما سمعتي الكلام عدل، قولي مثلاً:
"ها؟ عيدها شوي ما سمعتك عدل"
أو:
"شنو قلت؟ الصوت تقطع عندي"

إذا أحد يطقطق:
عادي ردي عليه بطقطقة خفيفة وبنفس جو الروم.

خلي نبرة الكلام حيوية وطبيعية وسريعة،
ولا تطولين في الرد إلا إذا السؤال يحتاج شرح.
"""


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger("K10_Voice")


# ============================================================
# CHECK ENV
# ============================================================

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN مو موجود")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY مو موجود")


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
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(
    command_prefix="!!",
    intents=intents,
    help_command=None
)

SESSIONS = {}


# ============================================================
# AUDIO HELPERS
# Discord:
# 48kHz / stereo / PCM16
#
# OpenAI:
# 24kHz / mono / PCM16
# ============================================================

def discord_to_live(pcm: bytes) -> bytes:
    """
    Discord 48k stereo PCM16
    ->
    OpenAI 24k mono PCM16
    """

    if not pcm:
        return b""

    try:
        audio = np.frombuffer(
            pcm,
            dtype=np.int16
        )

        if len(audio) < 2:
            return b""

        # Stereo -> Mono
        audio = audio[:len(audio) - (len(audio) % 2)]
        stereo = audio.reshape(-1, 2)

        mono = stereo.astype(
            np.int32
        ).mean(axis=1)

        mono = np.clip(
            mono,
            -32768,
            32767
        ).astype(np.int16)

        # 48k -> 24k
        mono_24k = resample_poly(
            mono,
            1,
            2
        )

        mono_24k = np.clip(
            mono_24k,
            -32768,
            32767
        ).astype(np.int16)

        return mono_24k.tobytes()

    except Exception as e:
        print(
            "❌ discord_to_live:",
            type(e).__name__,
            e,
            flush=True
        )

        return b""


def live_to_discord(pcm: bytes) -> bytes:
    """
    OpenAI 24k mono PCM16
    ->
    Discord 48k stereo PCM16
    """

    if not pcm:
        return b""

    try:
        audio = np.frombuffer(
            pcm,
            dtype=np.int16
        )

        if len(audio) == 0:
            return b""

        # 24k -> 48k
        audio_48k = resample_poly(
            audio,
            2,
            1
        )

        audio_48k = np.clip(
            audio_48k,
            -32768,
            32767
        ).astype(np.int16)

        # Mono -> Stereo
        stereo = np.column_stack(
            (audio_48k, audio_48k)
        ).reshape(-1)

        return stereo.astype(
            np.int16
        ).tobytes()

    except Exception as e:
        print(
            "❌ live_to_discord:",
            type(e).__name__,
            e,
            flush=True
        )

        return b""


# ============================================================
# DISCORD AUDIO SOURCE
# ============================================================

class K10AudioSource(discord.AudioSource):

    FRAME_SIZE = 3840

    def __init__(self):
        self.buffer = bytearray()
        self.lock = threading.Lock()

        # نخزن شوية صوت قبل التشغيل
        self.prebuffer_frames = 12
        self.started = False

    def put(self, data: bytes):

        if not data:
            return

        with self.lock:
            self.buffer.extend(data)

    def read(self):

        with self.lock:

            if not self.started:

                needed = (
                    self.FRAME_SIZE *
                    self.prebuffer_frames
                )

                if len(self.buffer) < needed:
                    return b"\x00" * self.FRAME_SIZE

                self.started = True

            if len(self.buffer) >= self.FRAME_SIZE:

                frame = bytes(
                    self.buffer[:self.FRAME_SIZE]
                )

                del self.buffer[:self.FRAME_SIZE]

                return frame

        return b"\x00" * self.FRAME_SIZE

    def is_opus(self):
        return False

    def cleanup(self):
        pass


# ============================================================
# K10 SESSION
# ============================================================

class K10Session:

    def __init__(self, guild_id, vc):

        self.guild_id = guild_id
        self.vc = vc

        self.connection = None

        self.main_task = None
        self.sender_task = None

        self.ready = asyncio.Event()
        self.started = asyncio.Event()

        self.start_error = None

        self.input_queue = asyncio.Queue()

        self.output = K10AudioSource()

        self.closed = False

        self.active_speaker = None
        self.last_speaker_time = 0

        self.audio_buffer = bytearray()


    # ========================================================
    # RECEIVE DISCORD AUDIO
    # ========================================================

    def push_discord_audio(
        self,
        user_id,
        pcm
    ):

        if self.closed:
            return

        if not pcm:
            return

        now = time.time()

        try:

            audio = np.frombuffer(
                pcm,
                dtype=np.int16
            )

            if len(audio) == 0:
                return

            level = float(
                np.abs(
                    audio.astype(np.int32)
                ).mean()
            )

            if level < INPUT_GATE:
                return

        except Exception:
            return

        # نخلي شخص واحد يتكلم في نفس الوقت
        if self.active_speaker is None:
            self.active_speaker = user_id

        elif self.active_speaker != user_id:

            if (
                now -
                self.last_speaker_time
                >
                SPEAKER_TIMEOUT
            ):
                self.active_speaker = user_id

            else:
                return

        self.last_speaker_time = now

        converted = discord_to_live(pcm)

        if not converted:
            return

        try:

            loop = bot.loop

            loop.call_soon_threadsafe(
                self.input_queue.put_nowait,
                converted
            )

        except Exception as e:

            print(
                "❌ QUEUE ERROR:",
                type(e).__name__,
                e,
                flush=True
            )


    # ========================================================
    # SEND AUDIO TO OPENAI
    # ========================================================

    async def send_audio(self):

        await self.started.wait()

        print(
            "🎤 Audio sender started",
            flush=True
        )

        pending = bytearray()

        while not self.closed:

            try:

                chunk = await self.input_queue.get()

                if not chunk:
                    continue

                pending.extend(chunk)

                while (
                    len(pending)
                    >=
                    LIVE_FRAME_BYTES
                ):

                    frame = bytes(
                        pending[:LIVE_FRAME_BYTES]
                    )

                    del pending[:LIVE_FRAME_BYTES]

                    if self.connection is None:
                        continue

                    encoded = base64.b64encode(
                        frame
                    ).decode("utf-8")

                    await (
                        self.connection
                        .input_audio_buffer
                        .append(
                            audio=encoded
                        )
                    )

            except asyncio.CancelledError:
                break

            except Exception as e:

                print(
                    "❌ SEND AUDIO ERROR:",
                    type(e).__name__,
                    e,
                    flush=True
                )

                await asyncio.sleep(0.1)


    # ========================================================
    # OPENAI REALTIME
    # ========================================================

    async def run(self):

        try:

            print(
                "🌐 Connecting to OpenAI Realtime...",
                flush=True
            )

            async with (
                openai_client
                .realtime
                .connect(
                    model=MODEL
                )
            ) as connection:

                self.connection = connection

                print(
                    "🌐 WebSocket connected",
                    flush=True
                )

                # إعداد الجلسة
                await connection.session.update(
                    session={
                        "type": "realtime",

                        "model": MODEL,

                        "instructions": SYSTEM_PROMPT,

                        "output_modalities": [
                            "audio"
                        ],

                        "audio": {

                            "input": {

                                "format": {
                                    "type": "audio/pcm",
                                    "rate": 24000
                                },

                                "turn_detection": {
                                    "type": "server_vad"
                                }
                            },

                            "output": {

                                "format": {
                                    "type": "audio/pcm",
                                    "rate": 24000
                                },

                                "voice": VOICE
                            }
                        }
                    }
                )

                print(
                    "📡 Session update sent",
                    flush=True
                )

                self.sender_task = asyncio.create_task(
                    self.send_audio()
                )

                async for event in connection:

                    event_type = event.type

                    # نطبع الأحداث المهمة
                    if event_type not in (
                        "response.output_audio.delta",
                    ):
                        print(
                            "📡 REALTIME EVENT:",
                            event_type,
                            flush=True
                        )

                    # ------------------------------
                    # SESSION CREATED
                    # ------------------------------

                    if event_type == "session.created":

                        print(
                            "✅ Realtime session created",
                            flush=True
                        )

                    # ------------------------------
                    # SESSION UPDATED = READY
                    # ------------------------------

                    elif event_type == "session.updated":

                        self.started.set()
                        self.ready.set()

                        print(
                            "✅ K10 REALTIME READY",
                            flush=True
                        )

                    # ------------------------------
                    # AUDIO FROM K10
                    # ------------------------------

                    elif (
                        event_type
                        ==
                        "response.output_audio.delta"
                    ):

                        try:

                            raw_audio = base64.b64decode(
                                event.delta
                            )

                            discord_audio = (
                                live_to_discord(
                                    raw_audio
                                )
                            )

                            self.output.put(
                                discord_audio
                            )

                        except Exception as e:

                            print(
                                "❌ OUTPUT AUDIO ERROR:",
                                type(e).__name__,
                                e,
                                flush=True
                            )

                    # ------------------------------
                    # K10 TRANSCRIPT
                    # ------------------------------

                    elif (
                        event_type
                        ==
                        "response.output_audio_transcript.delta"
                    ):

                        try:

                            print(
                                event.delta,
                                end="",
                                flush=True
                            )

                        except Exception:
                            pass

                    # ------------------------------
                    # USER STARTED TALKING
                    # ------------------------------

                    elif (
                        event_type
                        ==
                        "input_audio_buffer.speech_started"
                    ):

                        print(
                            "🎤 User started talking",
                            flush=True
                        )

                    # ------------------------------
                    # USER STOPPED TALKING
                    # ------------------------------

                    elif (
                        event_type
                        ==
                        "input_audio_buffer.speech_stopped"
                    ):

                        print(
                            "🛑 User stopped talking",
                            flush=True
                        )

                    # ------------------------------
                    # RESPONSE DONE
                    # ------------------------------

                    elif event_type == "response.done":

                        print(
                            "\n✅ Response done",
                            flush=True
                        )

                    # ------------------------------
                    # OPENAI ERROR
                    # ------------------------------

                    elif event_type == "error":

                        try:

                            error_message = (
                                event.error.message
                            )

                        except Exception:

                            error_message = str(
                                event
                            )

                        self.start_error = (
                            "OpenAI Realtime: "
                            + error_message
                        )

                        print(
                            "❌ OPENAI REALTIME ERROR:",
                            error_message,
                            flush=True
                        )

                        # إذا الخطأ صار قبل الجلسة
                        if not self.started.is_set():

                            self.ready.set()

                            break


        except asyncio.CancelledError:

            pass


        except Exception as e:

            self.start_error = (
                f"{type(e).__name__}: {e}"
            )

            print(
                "❌ LIVE ERROR:",
                self.start_error,
                flush=True
            )

            self.ready.set()


        finally:

            if (
                not self.ready.is_set()
            ):
                self.ready.set()

            if self.sender_task:

                self.sender_task.cancel()

            self.connection = None

            print(
                "🔌 Realtime session ended",
                flush=True
            )


    # ========================================================
    # CLOSE
    # ========================================================

    async def close(self):

        if self.closed:
            return

        self.closed = True

        print(
            "🧹 Closing K10 session...",
            flush=True
        )

        if self.sender_task:

            self.sender_task.cancel()

        if self.main_task:

            self.main_task.cancel()

        try:

            if (
                self.vc
                and
                self.vc.is_connected()
            ):

                try:
                    self.vc.stop()
                except Exception:
                    pass

                try:
                    self.vc.stop_listening()
                except Exception:
                    pass

                await self.vc.disconnect(
                    force=True
                )

        except Exception as e:

            print(
                "❌ CLOSE ERROR:",
                type(e).__name__,
                e,
                flush=True
            )


# ============================================================
# VOICE RECEIVE SINK
# ============================================================

class K10Sink(
    voice_recv.AudioSink
):

    def __init__(self, session):

        super().__init__()

        self.session = session


    def wants_opus(self):
        return True


    def write(
        self,
        user,
        data
    ):

        if user is None:
            return

        if getattr(
            user,
            "bot",
            False
        ):
            return

        try:

            pcm = data.pcm

            if not pcm:
                return

            self.session.push_discord_audio(
                user.id,
                pcm
            )

        except Exception as e:

            print(
                "❌ SINK ERROR:",
                type(e).__name__,
                e,
                flush=True
            )


    def cleanup(self):
        pass


# ============================================================
# BOT READY
# ============================================================

@bot.event
async def on_ready():

    print(
        "================================",
        flush=True
    )

    print(
        f"🟢 K10 Voice ONLINE: {bot.user}",
        flush=True
    )

    print(
        f"🤖 MODEL: {MODEL}",
        flush=True
    )

    print(
        f"🎙️ VOICE: {VOICE}",
        flush=True
    )

    print(
        "================================",
        flush=True
    )


# ============================================================
# JOIN
# ============================================================

@bot.command(
    name="join"
)
async def join(ctx):

    guild = ctx.guild

    if guild is None:
        return

    # لازم المستخدم يكون داخل روم
    if (
        ctx.author.voice is None
        or
        ctx.author.voice.channel is None
    ):

        await ctx.send(
            "❌ ادخل روم صوتي أول."
        )

        return

    channel = (
        ctx.author
        .voice
        .channel
    )

    await ctx.send(
        "🎙️ **K10 قاعد تدخل...**"
    )

    # ------------------------------
    # CLOSE OLD SESSION
    # ------------------------------

    old_session = SESSIONS.get(
        guild.id
    )

    if old_session:

        try:
            await old_session.close()
        except Exception:
            pass

        SESSIONS.pop(
            guild.id,
            None
        )

    # ------------------------------
    # OLD VOICE CLIENT
    # ------------------------------

    old_vc = guild.voice_client

    if old_vc:

        try:

            await old_vc.disconnect(
                force=True
            )

        except Exception:
            pass

        await asyncio.sleep(1)

    vc = None
    session = None

    try:

        print(
            f"🔊 Connecting Discord voice: {channel}",
            flush=True
        )

        vc = await channel.connect(
            cls=voice_recv.VoiceRecvClient,
            timeout=30.0,
            reconnect=True,
            self_deaf=False,
            self_mute=False
        )

        print(
            "✅ Discord voice connected",
            flush=True
        )

        # ------------------------------
        # CREATE K10 SESSION
        # ------------------------------

        session = K10Session(
            guild.id,
            vc
        )

        SESSIONS[guild.id] = session

        session.main_task = (
            asyncio.create_task(
                session.run()
            )
        )

        # ------------------------------
        # WAIT OPENAI
        # ------------------------------

        try:

            await asyncio.wait_for(
                session.ready.wait(),
                timeout=25
            )

        except asyncio.TimeoutError:

            raise RuntimeError(
                "OpenAI Realtime أخذ أكثر من 25 ثانية"
            )

        # ------------------------------
        # SHOW REAL ERROR
        # ------------------------------

        if not session.started.is_set():

            raise RuntimeError(
                session.start_error
                or
                "OpenAI Realtime ما بدأ الجلسة"
            )

        # ------------------------------
        # START DISCORD PLAYBACK
        # ------------------------------

        vc.play(
            session.output
        )

        # ------------------------------
        # START LISTENING
        # ------------------------------

        sink = K10Sink(
            session
        )

        vc.listen(
            sink
        )

        await ctx.send(
            "🟢 **K10 دخلت الروم وجاهزة 🎙️**"
        )

        print(
            "🎧 K10 listening...",
            flush=True
        )


    except Exception as e:

        error_text = (
            f"{type(e).__name__}: {e}"
        )

        print(
            "❌ JOIN ERROR:",
            error_text,
            flush=True
        )

        try:

            await ctx.send(
                "❌ **JOIN ERROR**\n"
                f"```{error_text[:1500]}```"
            )

        except Exception:
            pass

        if session:

            try:
                await session.close()
            except Exception:
                pass

        elif vc:

            try:

                await vc.disconnect(
                    force=True
                )

            except Exception:
                pass

        SESSIONS.pop(
            guild.id,
            None
        )


# ============================================================
# LEAVE
# ============================================================

@bot.command(
    name="leave"
)
async def leave(ctx):

    guild = ctx.guild

    if guild is None:
        return

    session = SESSIONS.pop(
        guild.id,
        None
    )

    if session:

        await session.close()

        await ctx.send(
            "👋 **K10 طلعت من الروم.**"
        )

        return

    vc = guild.voice_client

    if vc:

        try:

            await vc.disconnect(
                force=True
            )

        except Exception:
            pass

        await ctx.send(
            "👋 **طلعت من الروم.**"
        )

    else:

        await ctx.send(
            "❌ **أنا مب داخل روم أصلًا.**"
        )


# ============================================================
# STATUS
# ============================================================

@bot.command(
    name="status"
)
async def status(ctx):

    guild = ctx.guild

    if guild is None:
        return

    vc = guild.voice_client

    session = SESSIONS.get(
        guild.id
    )

    discord_status = (
        "🟢 متصل"
        if vc and vc.is_connected()
        else
        "🔴 مب متصل"
    )

    realtime_status = (
        "🟢 جاهز"
        if session and session.started.is_set()
        else
        "🔴 مب جاهز"
    )

    await ctx.send(
        "**K10 STATUS**\n"
        f"Discord Voice: {discord_status}\n"
        f"OpenAI Realtime: {realtime_status}\n"
        f"Model: `{MODEL}`"
    )


# ============================================================
# ERROR HANDLER
# ============================================================

@bot.event
async def on_command_error(
    ctx,
    error
):

    if isinstance(
        error,
        commands.CommandNotFound
    ):
        return

    print(
        "❌ COMMAND ERROR:",
        type(error).__name__,
        error,
        flush=True
    )


# ============================================================
# START
# ============================================================

print(
    "🚀 Starting K10 Voice...",
    flush=True
)

bot.run(
    DISCORD_TOKEN
)
