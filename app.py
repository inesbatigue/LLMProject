import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import ollama
import torch
import ast


def get_best_matching_text(question, df, model_name='paraphrase-MiniLM-L6-v2'):
    try:
        # Load the pre-trained model
        model = SentenceTransformer(model_name)

        # Extract texts and embeddings from the DataFrame
        texts = df['text'].tolist()
        embeddings = [torch.tensor(ast.literal_eval(embedding)) for embedding in df['embedding'].tolist()]

        # Generate the question embedding
        question_embedding = model.encode(question, convert_to_tensor=True)

        # Convert embeddings list to a tensor and move to the same device as question_embedding
        context_embeddings = torch.stack(embeddings).to(question_embedding.device)

        # Calculate similarities
        similarities = util.pytorch_cos_sim(question_embedding, context_embeddings)[0]

        # Find the best match
        best_match_idx = torch.argmax(similarities).item()
        best_match_score = similarities[best_match_idx].item()

        return texts[best_match_idx], best_match_score

    except Exception as e:
        print(f"Error in get_best_matching_text: {e}")
        return None
csv_file_path = 'embeddings.csv'
df = pd.read_csv(csv_file_path)
# Define the Streamlit interface
st.title("Question Answering with Llama3")

question = st.text_input("Enter your question:")
if question:
    # Assuming get_best_matching_text is defined elsewhere and available
    best_context, score = get_best_matching_text(question, df)
    ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

              ### Instruction:
              {}

              ### Input:
              {}

              ### Response:
              {}"""

    prompt = ALPACA_PROMPT.format(
        "Vous êtes un assistant intelligent. Vous recevrez en entrée une question sur le contexte fourni, puis vous y répondrez en français.",
        best_context,
        "",  # Leave this blank for generation
    )
    response = ollama.chat(model='llama3', messages=[{"role": "user", "content": prompt}])

    st.write("Response:")
    response_placeholder = st.empty()
    # Append each piece of the response to the output

    final_response = response['message']['content']
    response_placeholder.write(final_response)


    #st.write(final_response)



