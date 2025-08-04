import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Load your categories dataframe (ensure it's loaded with pandas)
# For example, you should have columns: 'category', 'subcategory', and 'ticket_id'
categories_df = pd.read_csv("path_to_your_categories_file.csv")

# Function to generate customer responses to agent prompts
def generate_customer_responses(agent_message, num_samples=500, max_length=50):
    prompt = f"Customer responds to: {agent_message}"  # Adjust the prompt to suit the task
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_texts = []

    for _ in range(num_samples):
        output_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1, num_beams=5, early_stopping=True)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_texts.append(generated_text)

    return generated_texts

# Dictionary to hold all responses per category/subcategory
all_responses = {}

# Iterate over unique categories and subcategories
for category in categories_df['category'].unique():
    for subcategory in categories_df[categories_df['category'] == category]['subcategory'].unique():
        
        # Get agent messages for each category/subcategory
        # You can adjust or generate different agent prompts based on category/subcategory
        agent_message = f"How can I assist you with your {subcategory} issue?"
        
        # Generate 500 customer responses
        responses = generate_customer_responses(agent_message, num_samples=500)
        
        # Store the responses grouped by category and subcategory
        if (category, subcategory) not in all_responses:
            all_responses[(category, subcategory)] = []
        
        all_responses[(category, subcategory)].extend(responses)

        print(f"Generated {len(responses)} responses for category '{category}' and subcategory '{subcategory}'")

# Save the generated responses to a file, including category and subcategory info
with open("categorized_customer_responses.txt", "w") as f:
    for (category, subcategory), responses in all_responses.items():
        for response in responses:
            f.write(f"Category: {category}, Subcategory: {subcategory}, Response: {response}\n")

print("Customer responses saved to 'categorized_customer_responses.txt'")