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

        # Add information about all available channels and their details
        if 'available_channels' in file_context and file_context['available_channels']:
            system_context += "\n\nAvailable channels and their details:"
            
            # If we have full channel details, use them
            if 'channel_details' in file_context:
                for channel, details in file_context['channel_details'].items():
                    unit = details.get('unit', 'N/A')
                    min_val = details.get('min_value', 'N/A')
                    max_val = details.get('max_value', 'N/A')
                    sampling_rate = details.get('sampling_rate', 'N/A')
                    description = details.get('description', 'N/A')
                    ecu = details.get('ecu', 'N/A')
                    data_type = details.get('data_type', 'N/A')
                    
                    system_context += f"\n- {channel}:"
                    system_context += f"\n  Unit: {unit}"
                    system_context += f"\n  Range: {min_val} to {max_val}"
                    system_context += f"\n  Sampling Rate: {sampling_rate}"
                    system_context += f"\n  Description: {description}"
                    system_context += f"\n  ECU: {ecu}"
                    system_context += f"\n  Data Type: {data_type}"
            # If we only have channel names (too many channels), show a warning
            elif file_context.get('channel_schema_only'):
                system_context += f"\n{file_context.get('channel_schema_warning', '')}"
                system_context += "\nChannel types:"
                for channel, data_type in file_context.get('channel_types', {}).items():
                    system_context += f"\n- {channel}: {data_type}"
            
            # Add selected channels if any
            if 'selected_channels' in file_context and file_context['selected_channels']:
                system_context += f"\n\nCurrently selected channels: {', '.join(file_context['selected_channels'])}"

        system_context += "\n\nYou can help analyze this automotive data, create visualizations, and find patterns or anomalies. Use the channel details above to provide accurate information about the signals."
        
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