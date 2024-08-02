# Final_project
This project aims to enhance the comprehension of complex legal documents using advanced large language models (LLMs). Legal documents, often characterized by dense and technical language, pose significant challenges for laypeople, leading to potential misunderstandings and compliance risks. This solution integrates sophisticated AI-driven tools to provide users with personalized, context-aware assistance, thereby improving their understanding of legal texts.

## Keyword Extraction
The Keyword Extraction component focuses on extracting relevant financial and legal keywords from Investopedia. By utilizing Selenium and Chromedriver, this module automates the retrieval of keyword definitions, examples, and usage contexts to build a comprehensive keyword dictionary that aids in understanding complex terms.

chromedriver/: Contains the necessary Chromedriver files for running automated browser tasks with Selenium.

Keyword.ipynb: A Jupyter Notebook that implements the keyword extraction logic and demonstrates how to use Selenium to extract keywords from Investopedia.

## User Personas
The project involves developing detailed user personas to tailor the platform's functionalities according to different literacy and numeracy levels. These personas guide the customization of the user interface and content delivery, ensuring that explanations and support are aligned with each user's comprehension capabilities.

## Chatbot
The project features an advanced chatbot that leverages keyword extraction, user personalization, and question-and-answer (Q&A) functionalities. This chatbot uses LLMs integrated with Retrieval-Augmented Generation (RAG) to offer real-time, tailored assistance.

Key Features:
Keyword Extraction: Identifies and explains key terms to enhance user comprehension.
User Personalization: Adapts responses based on user personas to provide relevant and accessible explanations.
Q&A Functionality: Allows users to ask specific questions about legal documents and receive accurate, context-aware answers.

## Flesch-Kincaid Readability Test
The Flesch-Kincaid readability test to assess the readability of the explanations provided by the chatbot. This test calculates the Flesch-Kincaid score for both the keyword explanations and the Q&A responses, ensuring that the content is appropriately tailored to different user personas.
