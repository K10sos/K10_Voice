import asyncio
import discord


def setup_voice_commands(bot):

    @bot.command(name="join")
    async def join_voice(ctx):

        if not ctx.author.voice or not ctx.author.voice.channel:
            await ctx.send("❌ ادخل روم صوتي أول.")
            return

        channel = ctx.author.voice.channel

        try:
            # إذا فيه اتصال صوتي قديم
            if ctx.voice_client:

                if ctx.voice_client.is_connected():

                    if ctx.voice_client.channel.id == channel.id:
                        await ctx.send("🎙️ أنا موجود معاك في الروم أصلًا.")
                        return

                    await ctx.voice_client.move_to(channel)
                    await ctx.send(f"✅ انتقلت إلى **{channel.name}**")
                    return

                else:
                    await ctx.voice_client.disconnect(force=True)

            # اتصال جديد
            voice_client = await channel.connect(
                timeout=30.0,
                reconnect=True,
                self_deaf=False,
                self_mute=False
            )

            if voice_client and voice_client.is_connected():
                await ctx.send(f"🟢 دخلت الروم الصوتي **{channel.name}**")
            else:
                await ctx.send("❌ الاتصال بالروم ما اكتمل.")

        except asyncio.TimeoutError:
            print("[VOICE ERROR] Voice connection timed out")

            try:
                if ctx.voice_client:
                    await ctx.voice_client.disconnect(force=True)
            except:
                pass

            await ctx.send(
                "❌ VOICE ERROR:\n"
                "```TimeoutError: Discord Voice connection timed out```"
            )

        except discord.Forbidden:
            await ctx.send(
                "❌ ما عندي صلاحية أدخل الروم.\n"
                "تأكد من View Channel + Connect + Speak."
            )

        except Exception as e:
            print(f"[VOICE JOIN ERROR] {type(e).__name__}: {e}")

            try:
                if ctx.voice_client:
                    await ctx.voice_client.disconnect(force=True)
            except:
                pass

            await ctx.send(
                f"❌ VOICE ERROR:\n```{type(e).__name__}: {e}```"
            )


    @bot.command(name="leave")
    async def leave_voice(ctx):

        try:
            if not ctx.voice_client:
                await ctx.send("🔇 أنا مب داخل روم صوتي.")
                return

            await ctx.voice_client.disconnect(force=True)
            await ctx.send("👋 طلعت من الروم.")

        except Exception as e:
            print(f"[VOICE LEAVE ERROR] {type(e).__name__}: {e}")
            await ctx.send(
                f"❌ LEAVE ERROR:\n```{type(e).__name__}: {e}```"
            )