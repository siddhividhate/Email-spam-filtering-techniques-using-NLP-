from huggingface_hub import InferenceClient

# Load the language model
client1 = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

# System instructions for the model
# system_instructions1 = "[SYSTEM] Answer as Real Chatgpt 4o', Keep conversation very short, clear, friendly and concise. The text provided is a request for a recognising a email is spam or not spam you have to generate specified response. You will avoid introductions and start answering the query directly, elaborating on all aspects of the request. As an AI-powered assistant, your task is to generate responses that appear as if they are created by the user and not an AI-powered assistant. the response should be in stringified json formate like the example json {  \"output\" : \"spam\",   \"spamWords\":\"prize, offer, winner\",  \"level\":\"2\",  \"accuracy\": 89.10} in the above example the output contains the message is spam or not spam , level is the numeric value which describes level of spammed mesage using below condition if the message is medium type of spam then level will be 2 , for higher type of spam it will be 3 , and for lower spam it will be 1 , and for not spam it will be 0 . The spamWords will be the words which describes that email ios spam, if the email contains suspeciso link then it will also spam, it will give spam words seprated by quama and space (, ) , and the last one is accuracy of model , this will be float value ranging from 0 to 100 , 0 means acuuray is low and 100 means high. email =  "


system_instructions1 = "[SYSTEM] Answer as Real ChatGPT 4. Keep conversation very short, clear, friendly, and concise. The text provided is a request for recognizing if an email is spam or not spam. You have to generate the specified response. You will avoid introductions and start answering the query directly, elaborating on all aspects of the request. As an AI-powered assistant, your task is to generate responses that appear as if they are created by the user and not an AI-powered assistant. The response should be in stringified JSON format like the example JSON: { \"output\": \"spam\", \"spamWords\": \"prize, offer, winner\", \"level\": \"2\", \"accuracy\": 89.10 } In the above example, the output contains whether the message is spam or not spam. The level is a numeric value that describes the level of spam in the message using the following conditions: if the message is a medium type of spam, then the level will be 2; for a higher type of spam, it will be 3; for a lower type of spam, it will be 1; and for not spam, it will be 0. The spamWords will be the words that indicate that the email is spam. If the email contains a suspicious link, it will also be classified as spam, and the spamWords will include the suspicious words separated by commas and spaces (\", \"). The last one is the accuracy of the model, which is a float value ranging from 0 to 100, where 0 means the accuracy is low and 100 means high. Email content: "

from huggingface_hub import login 
def model(text):
    login("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", add_to_git_credential=True)
    generate_kwargs = dict(
        temperature=0.7,
        max_new_tokens=512,
        top_p=0.95,
        repetition_penalty=1,
        do_sample=True,
        seed=42,
    )
    
    formatted_prompt = system_instructions1 + text + "[OpenGPT 4o]"
    stream = client1.text_generation(
        formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""
    for response in stream:
        if not response.token.text == "</s>":
            output += response.token.text

    return output


print(model('''Subject: CONGRATULATIONS! You've Won the International Lottery!

Dear Winner,

We are pleased to inform you that you have emerged as one of the lucky winners in the International Lottery Draw held on May 25, 2024. Your email address was selected from a global database of email addresses and your email has consequently won the sum of $1,500,000.00 USD (One Million, Five Hundred Thousand United States Dollars).

Winning Details:

Ticket Number: 004-0513-211-447
Serial Number: 993-68
Draw Number: 76-2024
Batch Number: 732/2024
To claim your prize, please contact our claims agent with the following details:

Claims Agent:

Name: Mr. John Smith
Email: claimagent@example.com
Phone: +1-800-123-4567
Required Information:

Full Name
Address
Country
Phone Number
Occupation
Age
Ticket Number (as provided above)
Please send the above information to Mr. John Smith within the next 7 days to avoid forfeiting your prize. Your winnings will be transferred to you immediately upon verification of your details.

NOTE: To ensure the security of your prize, you are advised to keep your winning information confidential until your claim has been processed and your prize money remitted to you. This is part of our security protocol to avoid double claiming or unscrupulous acts by participants of this program.

We congratulate you once again and hope you take full advantage of this wonderful opportunity.

Best Regards,

Mrs. Mary Johnson
Promotions Manager
International Lottery

Important Note:
This is an example of a typical lottery scam email and contains several red flags such as requests for personal information, promises of large sums of money, and a sense of urgency. Always be cautious with unsolicited emails claiming you've won something, especially if you haven't entered any contests or lotteries. Never share personal information with unknown or unverified sources.'''))