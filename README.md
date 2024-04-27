# AutoML_GPT

#### Description: 
This project aims to simplify the process of machine learning model creation, training, and evaluation through an intuitive conversational interface. Leveraging the power of ChatGPT's language understanding capabilities and the LangChain's framwork to tune ChatGPT on our machine learning documentation. The AutoML GPT assistant allows users to interactively design, train, and evaluate machine learning models using custom documentation. The project provides a seamless experience for users to specify model configurations, perform hyperparameter tuning, and analyze model performance through natural language queries.

#### Below is the workflow of the project : 
<img width="735" alt="Workflow Diagram" src="https://github.com/pparashar21/AutoML_GPT/assets/103917169/716e13ad-b326-40b0-9ad8-c2d48a2ef80a">

#### Working: 
1. Engage with the chatbot by posing inquiries about various machine learning models and their parameters
<img width="735" alt="pic1" src="https://github.com/pparashar21/AutoML_GPT/assets/103917169/691f1cf4-c11b-487d-bfef-88ee327c9466">
<img width="735" alt="pic2" src="https://github.com/pparashar21/AutoML_GPT/assets/103917169/244acdce-9edb-4f1e-bf89-ae26ad4b599e">

2. Easily configure your model and input dataset specifications, seamlessly followed by training and evaluation

- Perform model training with specified parameter values (no hyperparameter tuning)
<img width="735" alt="pic3" src="https://github.com/pparashar21/AutoML_GPT/assets/103917169/8650b328-248f-4c95-b701-a616572ef429">
<img width="735" alt="pic4" src="https://github.com/pparashar21/AutoML_GPT/assets/103917169/282cb834-1fa4-470a-82b2-227e2dfe06ee">

- Perform hyperparameter tuning by specifying multiple values for a parameter (utilizes GridSearchCV)
<img width="735" alt="pic5" src="https://github.com/pparashar21/AutoML_GPT/assets/103917169/889a79f4-724a-480a-958f-9703c6449ba3">

3. Compare different models on their performance to determine best model and their configurations
<img width="741" alt="pic6" src="https://github.com/pparashar21/AutoML_GPT/assets/103917169/a3dae713-4d89-436e-9b52-b8e0362786d1">

#### Usage:
- Add the datasets you want to run the model on in the "datasets" folder. Please note, this project will not perform any type of data preprocessing, only model training and evaluation.
- Once the script is running, an input box will prompt you to enter your question or command.
- Type your queries about machine learning models, parameters, dataset details, or any other related topic.
- Follow the prompts provided by the chatbot to specify model configurations, perform training and evaluation, and analyze results.
- To exit the chatbot, type 'exit'.
