from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint

from core import get_model, settings


@entrypoint(checkpointer=MemorySaver())
async def automotive_agent(
    inputs: dict[str, list[BaseMessage]],
    *,
    previous: dict[str, list[BaseMessage]],
    config: RunnableConfig,
):
    """
    An agent specifically designed for automotive data analysis.
    
    This agent can:
    - Understand and analyze automotive MDF files
    - Process queries about automotive signals/channels
    - Generate visualizations and statistics about automotive data
    """
    messages = inputs["messages"]
    if previous:
        messages = previous["messages"] + messages

    # Get any file context from the config
    file_context = config["configurable"].get("current_file", {})
    
    # Add file context to the system prompt if available
    if file_context:
        # Prepare a detailed system prompt about the automotive data
        system_context = f"""You are an automotive data analysis assistant specialized in MDF file analysis.
        
Current file: {file_context.get('filename', 'No file selected')}
File type: {file_context.get('file_type', 'Unknown')}
Duration: {file_context.get('duration', 0)} seconds
Channel count: {file_context.get('channel_count', 0)}
"""

        # Add information about selected channels if available
        if 'selected_channels' in file_context and file_context['selected_channels']:
            system_context += f"\nSelected channels: {', '.join(file_context['selected_channels'])}"
            
            # Add channel details if available
            if 'channel_details' in file_context:
                system_context += "\n\nChannel details:"
                for channel, details in file_context.get('channel_details', {}).items():
                    unit = details.get('unit', 'N/A')
                    min_val = details.get('min_value', 'N/A')
                    max_val = details.get('max_value', 'N/A')
                    system_context += f"\n- {channel}: Unit: {unit}, Range: {min_val} to {max_val}"
        
        # Add information about available channels if no specific channels are selected
        elif 'available_channels' in file_context and file_context['available_channels']:
            sample_channels = file_context['available_channels'][:5]  # Show up to 5 channels
            system_context += f"\n\nSample available channels: {', '.join(sample_channels)}"

        system_context += "\n\nYou can help analyze this automotive data, create visualizations, and find patterns or anomalies."
        
        # Prepend the system message to the conversation
        from langchain_core.messages import SystemMessage
        system_message = SystemMessage(content=system_context)
        
        # Add system message only if it doesn't already exist
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = [system_message] + messages

    # Get the model from the config
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    
    # Generate response
    response = await model.ainvoke(messages)
    
    return entrypoint.final(
        value={"messages": [response]}, save={"messages": messages + [response]}
    ) 