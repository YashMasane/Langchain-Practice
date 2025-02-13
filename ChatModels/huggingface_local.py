from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=1.2,
        max_new_tokens=100

    )
)

model = ChatHuggingFace(llm=llm)

result = model.invoke('What is capital of india')

print(result.content)