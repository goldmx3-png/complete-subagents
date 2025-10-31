Okay now this /mnt/d/chatbot-research/complete-subagents project is a copy of /mnt/d/chatbot-research/complete-agent. 

now few things are already implimented from complete-agent project. I want you to continue developing the rest of the project. Here are the guidelines:

- You can refer the complete-agent code.
- Do not copy complete things if it is not correct practice


My Plan:
- If user clicks buttons maintain a flag to identify the button clicks and respond quickly.
    i'll guide in future to develop the full menu.
- If user is asking direct questions from text box classify if it is asking realtime banking data like live data or just policies. classify based on user query and route to rag_agent or api_agent. also if it is api_agent identify the product and in next step follow same as complete-agent code. (right no in complete-agent we are using different llm calls for classifying rag or api and then calling different llm call for api selection. here we merge this llm call.)
- Also we need to we care about follow up question. Impliment this as industry standard for Langchain best practices.
- Rest of the things you refer complete-agent code.
