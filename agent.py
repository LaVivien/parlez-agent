#https://docs.livekit.io/agents/voice-agent/voice-pipeline/
#from the template
import logging

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, openai, deepgram, silero, turn_detector, elevenlabs, google

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "Your name is Jack. You are 36 years old. You can speak French. You are a French tutor."
            "you help English speaking people to practice French conversation. Their French levels are A1, A2, B1, B2. "
            "Use simple, short, conversational responses, ask back questions to continue practicing French. "
            "don't use more than 4 sentences in your answer. don't use unpronouncable punctuation or moji."
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    # This project is configured to use Deepgram STT, OpenAI LLM and Cartesia TTS plugins
    # Other great providers exist like Cerebras, ElevenLabs, Groq, Play.ht, Rime, and more
    # Learn more and pick the best one for your app:
    # https://docs.livekit.io/agents/plugins
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],

        #STT  
        #stt=openai.STT(language="fr",  model="whisper-1"),
        stt=openai.STT.with_groq(model="whisper-large-v3"),

        #LLM
        #llm=openai.LLM(model="gpt-4o-mini"),
        llm=openai.LLM.with_groq(model="gemma2-9b-it"),

        #TTS
        tts=openai.TTS(voice="echo"), 

        #tts=elevenlabs.TTS(voice=elevenlabs.tts.Voice( id="pNInz6obpgDQGcFmaJgB", name="Adam",\
        #tts=elevenlabs.TTS(voice=elevenlabs.tts.Voice( id="EXAVITQu4vr4xnSDxMaL", name="Bella",                                              
        #category="premade"), model="eleven_multilingual_v2"),

        turn_detector=turn_detector.EOUModel(),
        # minimum delay for endpointing, used when turn detector believes the user is done with their turn
        min_endpointing_delay=0.5,
        # maximum delay for endpointing, used when turn detector does not believe the user is done with their turn
        max_endpointing_delay=5.0,
        chat_ctx=initial_ctx,
    )

    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    agent.start(ctx.room, participant)

    # The agent should be polite and greet the user when it joins :)
    await agent.say("Bonjour, tu veux pratiquer le français?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
