**Title: AI Chatbot with LLM**

**Summary/ Abstract:**

The project addresses the need for an AI chatbot capable of efficiently retrieving relevant information from large datasets using natural language queries. Leveraging Pinecone's vector search engine and OpenAI's language models, Langchain and Streamlit, we aim to develop a chatbot that provides accurate and contextually relevant responses to user queries. The project focuses on enabling users to search through YouTube encyclopedia using natural language, enhancing accessibility and searchability of video content.

**Problem Statement:**

Despite significant investment in internal support infrastructure, companies still struggle with inefficiencies and delays in resolving employee queries. Traditional support channels like email often lead to bottlenecks and decreased productivity. Complexity in internal processes further hampers timely and accurate support provision. Customized chatbot solutions are increasingly essential to address these challenges, delivering instant assistance, automating tasks, and streamlining workflows. They seamlessly integrate with existing systems, providing valuable support across company processes.

**Dataset Source:**

The dataset, known as "YouTube Wikipedia," is a publicly available encyclopedia on YouTube accessible via the following link: https://en.wikipedia.org/wiki/YouTube. For demonstration purposes, we acquired the encyclopedia in PDF format, segmented it into chunks using `RecursiveCharacterTextSplitter`, and subsequently uploaded all these chunks into the Pinecone database using the `text-embedding-ada-002` model from the OpenAPI LLM. The entire document comprises 166 vectors, each representing a chunk detail are elaborated in the report.

**Technology/Feature Demonstrated:**

The key technologies demonstrated include: 

•	Pinecone DB efficiently store, retrieve, and analyze vast amounts of data.

•	OpenAI's LLM extract meaningful insights and perform precise text embeddings.

•	Langchain enhances the bot's context memory and response personalization.

•	Streamlit for building the chat interface.

The integration of these technologies enables efficient retrieval of semantically relevant information from the dataset, enhancing the chatbot's performance and user experience.

**Uses:**

The AI chatbot facilitates easy access to information within YouTube encyclopedia, allowing users to search for specific content or topics using natural language queries. It enhances accessibility for individuals with hearing impairments, improves search engine optimization (SEO) for video content, and aids in better comprehension of video content.

**Benefits:**

•	Efficient retrieval of relevant information from large datasets

•	Enhanced accessibility and searchability of video content

•	User-friendly interface for seamless interaction

•	Accurate and contextually relevant responses to user queries

**Drawbacks:**

•	Dependency on the quality and coverage of the dataset.

•	Potential biases in the data that may affect response accuracy.

•	Occasional delays in response time, particularly for complex queries.

**Challenges:**

•	Optimizing performance and accuracy of the chatbot

•	Mitigating bias in the training data and responses

•	Minimizing response time delays for improved user experience

**Results:**

Our testing efforts confirm the successful performance of the AI chatbot in providing relevant responses to user queries. The integration of Pinecone and OpenAI's language models contributes to the efficiency and accuracy of information retrieval. The user-friendly chat interface built using Streamlit enhances user engagement and satisfaction.

**Working Example:**

In the working example, users interact with the chatbot via Streamlit, inputting natural language queries related to YouTube encyclopedia.

<img width="245" alt="image" src="https://github.com/user-attachments/assets/c52d6f42-cab0-4cd8-af05-235de8dd3447">
 
**Author: Seema Kavatkar**

