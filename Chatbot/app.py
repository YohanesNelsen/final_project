import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import spacy
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain import PromptTemplate, LLMChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Predefined topics
unique_topics = ['capital', 'capital asset pricing model', 'capital expenditure', 'capitalism', 'central limit theorem', 'chartered financial analyst', 'chief executive officer', 
                  'code of ethics', 'coefficient of variation', 'collateral', 'command economy', 'comparative advantage', 'compound annual growth rate', 'compound interest', 'conflict theory', 
                  'consumer price index', 'contribution margin', 'correlation', 'correlation coefficient', 'cost of goods sold', 'creative destruction', 'credit default swap', 'current ratio', 
                  'customer service', 'absolute advantage', 'accounting equation', 'accounting rate of return', 'acid-test ratio', 'acquisition', 'adverse selection', 'after-hours trading', 
                  'alpha', 'amalgamation', 'american depositary receipt', 'american dream', 'analysis of variance', 'angel investor', 'annual percentage rate', 'annuity', 'applicable federal rate', 
                  'artificial intelligence', 'asset', 'asset management', 'asset turnover ratio definition', 'assets under management', 'automated clearing house', 'automated teller machine', 'average true range', 
                  'balanced scorecard', 'balance sheet', 'bank identification numbers', 'bankruptcy', "baye's theorem", 'bear market', 'berkshire hathaway', 'bernie madoff', 'beta', 'bill of lading', 'bitcoin mining', 
                  'blockchain', 'bollinger band', 'bond', 'break-even analysis', 'brexit', 'budget', 'budget deficit', 'bull market', 'business cycle', 'business ethics', 'business model', 'business-to-consumer', 
                  'business valuation', 'xml', 'days payable outstanding', 'days sales outstanding', 'debenture', 'debt ratio', 'debt-service coverage ratio', 'debt-to-equity ratio', 'deferred compensation', 
                  'delivered-at-place', 'delivered duty paid', 'delivered duty unpaid', 'demand', 'demonetization', 'derivative', 'dilution', 'disbursement', 'discount rate', 'diversification', 'dividend', 
                  'dividend payout ratio', 'dividend yield', 'dow jones industrial average', 'due diligence', 'dupont analysis', 'demand elasticity', 'p-value', 'partnership', 'penny stocks trade', 'per capita gdp', 
                  'perfect competition', 'what is personal finance, and why is it important?', 'phillips curve', 'ponzi schemes: definition, examples, and origins', "porter's 5 forces", 'positive correlation', 'pre-market', 
                  'preference shares', 'preferred stock', 'present value', 'price-to-earnings ratio', 'price/earnings-to-growth ratio', 'pro rata', 'producer price index', 'profit', 'profit and loss statement', 'promissory note', 
                  'prospectus', 'public limited company', 'put option', 'earnest money', 'earnings before interest and taxes', 'earnings before interest, taxes, depreciation and amortization', 'earnings per share', 'ebita', 
                  'economic growth', 'economic moat', 'economics', 'economies of scale', 'employee stock ownership plan', 'endowment fund', 'enterprise resource planning', 'enterprise value', 'entrepreneur', 'environmental protection agency', 
                  'environmental, social, and governance criteria', 'equity', 'equivalent annual cost', 'escrow', 'european union', 'ex-dividend', 'exchange rate', 'exchange-traded fund', 'externality', 'faang stocks', 
                  'factors of production', 'fang stocks', 'feasibility study', 'federal deposit insurance corporation', 'federal funds rate', 'federal housing administration loan', 'federal insurance contributions act', 
                  'fiat money', 'fiduciary', 'finance', 'financial institution', 'financial statements', 'financial technology', 'fiscal policy', 'fixed income', 'fixed-income security', 'four ps', 'free carrier', 'free market', 'free on board', 
                  'free trade', 'fringe benefits', 'futures', 'game theory', 'gamma', 'general agreement on tariffs and trade', 'general data protection regulation', 'general ledger', 'generally accepted accounting principles', 'generation x', 
                  'geometric mean', 'giffen good', 'gini index', 'globalization', 'goods and services tax', 'goodwill', 'gordon growth model', 'government bond', 'government shutdown', 'great depression', 'gross domestic product', 'gross income', 
                  'gross margin', 'gross national product', 'gross profit', 'gross profit margin', 'guarantor', 'hard skills', 'harmonic mean', 'head and shoulders pattern', 'health maintenance organizations', 'health savings account', 
                  'hedge', 'hedge fund', 'herfindahl-hirschman index', 'heteroskedasticity', 'high-low method', 'high-net-worth individual', 'hold harmless clause', 'holding company', 'home equity loan', 'homeowners association', 'homeowners association fee', 
                  'homestead exemption', 'horizontal integration', 'hostile takeover', 'housing bubble', 'human capital', 'hurdle rate', 'hyperinflation', 'hypothesis testing', 'income', 'income statement', 'indemnity', 'indemnity insurance', 'index fund', 
                  'individual retirement account', 'industrial revolution', 'inferior good', 'inflation', 'initial public offerings', 'insider trading', 'insurance', 'insurance premium', 'interest coverage ratio', 'interest rate', 'internal rate of return', 
                  'internal revenue service', 'international financial reporting standards', 'international monetary fund', 'interpersonal skills', 'inventory turnover', 'inverted yield curve', 'invisible hand', 'irrevocable trust', 'j-curve', 'january effect', 
                  'japanese government bond', "jensen's measure", 'job market', 'john maynard keynes', 'joint and several liability', 'joint probability', 'joint-stock company', 'joint tenancy', 'joint venture', 'jones act', 'joseph schumpeter', 'journal', 
                  'joint tenants with right of survivorship', 'jpy', 'jumbo cd', 'jumbo loan', 'junk bond', 'juris doctor', 'jurisdiction risk', 'just in case', 'just in time', 'kaizen', 'karl marx', 'keltner channel', 'keogh plan', 'key performance indicators', 
                  'key person insurance', 'key rate duration', 'keynesian economics', 'kickback', 'kiddie tax', "kids in parents' pockets eroding retirement savings", 'kiosk', 'kiting', 'klinger oscillator', 'knock-in option', 'knock-out option', 'know sure thing', 
                  'know your client', 'knowledge economy', 'knowledge process outsourcing', 'korean composite stock price indexes', 'kurtosis', 'kuwaiti dinar', 'the kyoto protocol', 'laissez-faire', 'law of demand', 'law of supply', 'law of supply and demand', 
                  'leadership', 'letter of intent', 'letters of credit', 'leverage', 'leverage ratio', 'leveraged buyout', 'liability', 'liability insurance', 'limit order', 'what is a limited government, and how does it work?', 'limited liability company', 
                  'limited partnership', 'line of credit', 'liquidation', 'liquidity', 'liquidity coverage ratio', 'liquidity ratio', 'loan-to-value ratio', 'london inter-bank offered rate', 'ltd.', 'macroeconomics', 'magna cum laude', 'management by objectives', 
                  'margin', 'margin call', 'market share', 'marketing', 'marketing strategy', 'master limited partnership', 'memorandum of understanding', 'mercantilism', 'mergers and acquisitions', 'milton friedman', 'mixed economic system', 'monetary policy', 
                  'money laundering', 'money market account', 'monopolistic competition', 'monte carlo simulation', "moore's law", 'moving average convergence divergence', 'multilevel marketing', 'mutual fund', 'mutually exclusive', 'nasdaq', 'nash equilibrium', 
                  'negative correlation', 'neoliberalism', 'net asset value', 'net income', 'net operating income', 'net present value', 'net profit margin', 'net worth', 'netting', 'network marketing', 'networking', 'new york stock exchange', 'next of kin', 
                  'ninja loan', 'nominal', 'non-disclosure agreement', 'normal distribution', 'north american free trade agreement', 'not for profit', 'notional value', 'novation', 'null hypothesis', 'offset', 'old age, survivors, and disability insurance', 
                  'oligopoly', 'onerous contract', 'online banking', 'open market operations', 'operating income', 'operating leverage', 'operating margin', 'operations management', 'opportunity cost', 'option', 'organization of the petroleum exporting countries', 
                  'organizational behavior', 'organizational structure', 'original equipment manufacturer', 'original issue discount', 'out of the money', 'outsourcing', 'over-the-counter', 'over-the-counter market', 'overdraft', 'overhead', 'overnight index swap', 
                  'q ratio', 'quadruple witching', 'qualified dividend', 'qualified institutional buyer', 'qualified institutional placement', 'qualified longevity annuity contract', 'qualified opinion', 'qualified retirement plan', 'qualified terminable interest property trust', 
                  'qualitative analysis', 'quality control', 'quality of earnings', 'quality management', 'quantitative analysis', 'quantitative easing', 'quantitative trading', 'quantity demanded', 'quarter', 'quarter on quarter', 'quasi contract', 'quick assets', 
                  'quick ratio', 'quintiles', 'quota', 'r-squared', 'racketeering', 'rate of return', 'rational choice theory', 'real estate', 'real estate investment trust', 'real gross domestic product', 'receivables turnover ratio', 'registered investment advisor', 
                  'regression', 'relative strength index', 'renewable resource', 'repurchase agreement', 'requests for proposal', 'required minimum distribution', 'retained earnings', 'return on assets', 'return on equity', 'return on invested capital', 'return on investment', 
                  'roth 401', 'roth ira', 'rule of 72', 'russell 2000 index', 's&p 500 index', 'sarbanes-oxley act of 2002', 'securities and exchange commission', 'security', 'series 63', 'series 7', 'sharpe ratio', 'short selling', 'social media', 'social responsibility', 
                  'solvency ratio', 'spread', 'standard deviation', 'stochastic oscillator', 'stock', 'stock keeping unit', 'stock market', 'stop-limit order', 'straddle', 'strength, weakness, opportunity, and threat analysis', 'subsidiary', 'supply chain', 'sustainability', 
                  'systematic sampling', 't-test', 'tariff', 'technical analysis', 'tenancy in common', 'term life insurance', 'terminal value', 'third world', 'total-debt-to-total-assets', 'total expense ratio', 'total quality management', 'total shareholder return', 
                  'trade deficit', 'trailing 12 months', 'tranches', 'transaction', 'treasury bills', 'treasury inflation-protected security', 'triple bottom line', 'troubled asset relief program', 'trust', 'trust fund', 'trustee', 'tsa precheck', 'turnover', 
                  'underlying asset', 'underwriter', 'underwriting', 'unearned income', 'unemployment', 'unemployment rate', 'unicorn', 'unified managed account', 'uniform distribution', 'uniform gifts to minors ac', 'uniform transfers to minors act', 'unilateral contract', 
                  'unit investment trust', 'united nations', 'universal life insurance', 'unlevered beta', 'unlevered free cash flow', 'unlimited liability', 'unsecured loan', 'upside', 'u.s. dollar index', 'u.s. savings bonds', 'utilities sector', 'utility', 'valuation', 
                  'value added', 'value chain', 'value investing', 'value proposition', 'value at risk', 'value-added tax', 'variability', 'variable annuity', 'variable cost', 'variance', 'vega', 'velocity of money', 'venture capital', 'venture capitalist', 'vertical analysis', 
                  'vertical integration', 'vesting', 'visual basic for applications', 'vix', 'volatility', 'volcker rule', 'volume weighted average price', 'voluntary employees beneficiary association plan', 'w-2 form', 'w-4 form', 'w-8 form', 'waiver of subrogation', 
                  'wall street', 'war bond', 'warrant', 'wash sale', 'wash-sale rule', 'wealth management', 'wearable technology', 'weighted average', 'weighted average cost of capital', 'white-collar crime', 'white paper', 'wholesale price index', 'wire fraud', 
                  'wire transfers', 'withholding allowance', 'withholding tax', 'working capital', 'works-in-progress', 'world trade organization', 'worldcom', 'x-efficiency', 'x-mark signature', 'xbrl', 'xcd', 'xd', 'xenocurrency', 'xetra', 'xrt', 
                  'yacht insurance', 'yale school of management', 'yankee bond', 'yankee market', 'year-end bonus', 'year-over-year', 'year to date', "year's maximum pensionable earnings", 'yearly rate of return method', 'yearly renewable term', 'yield',
                  'yield basis', 'yield curve', 'yield curve risk', 'yield maintenance', 'yield on cost', 'yield on earning assets', 'yield spread', 'yield to call', 'yield to maturity', 'yield to worst', 'yield variance', 'york antwerp rules', 'yuppie', 'z-score', 
                  'z-test', 'zacks investment research', 'zcash', 'zero balance account', 'zero-based budgeting', 'zero-beta portfolio', 'zero-bound', 'zero coupon inflation swap', 'zero coupon swap', 'zero cost collar', 'zero-coupon bond', 'zero-lot-line house', 
                  'zero-one integer programming', 'zero-rated goods', 'zero-sum game', 'zero-volatility spread', 'zeta model', 'zig zag indicator', 'zk-snark', 'zombies', 'zoning', 'zoning ordinance', 'zzzz best']

def preprocess_text(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    text = re.sub(r'[^a-zA-Z0-9\s.,\'-]', '', text)
    text = re.sub(' +', ' ', text).strip().lower()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
    filtered_tokens = [token for token in lemmatized_tokens if token not in stop_words or token == 'not']
    return ' '.join(filtered_tokens)

def find_matching_topics(input_string, topics):
    input_string = input_string.lower()
    found_topics = [topic for topic in topics if topic in input_string]
    
    return found_topics

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4")
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if 'conversation' in st.session_state and callable(st.session_state.conversation):
        knowledge_level = st.session_state.knowledge_level
        
        # Define knowledge level context
        if knowledge_level == "Beginner":
            context_description = (
                "Use words that are used by elementary students. The explanation should be for someone with a low to mid income (quintile 1-3) "
                "and primary to lower secondary education, around the level of 7 and 9 grader with low literacy and numeracy scores. "
                "Keep the sentence short and use one or two-syllable words."
                "If any financial term is mentioned, also briefly explain related concepts."
                "Answer in 2 paragraph maximum"
            )
        elif knowledge_level == "Intermediate":
            context_description = (
                "The explanation should be for someone with a low to mid income (quintile 1-3) "
                "and upper secondary to post-secondary non-tertiary education (ISCED 3C - ISCED 4C). "
                "Assume medium literacy and numeracy scores. "
                "Use words with 1-3 syllables."
                "If any financial term is mentioned, also briefly explain related concepts."
                "Answer in 2 paragraph maximum"
            )
        else:  # Advanced
            context_description = (
                "The explanation should be for someone with a mid to high income (quintile 4-5) "
                "and tertiary education (ISCED 5 and above). "
                "Assume high literacy and numeracy scores. "
                "Use precise and accurate terminology, but ensure clarity."
                "If any financial term is mentioned, also briefly explain related concepts."
                "Answer in 2 paragraph maximum"
            )

        # Construct the modified question with additional context
        modified_question = (
            f"{user_question}. If this question is not related to financial/law, say 'I'm sorry, I don't have the information you need. "
            f"Also give a short example if possible. Please explain in a way suitable for a {knowledge_level} level. {context_description}"
        )
        
        # Get the response from the conversation chain
        response = st.session_state.conversation({'question': modified_question})
        
        # Add the user question and response to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})

        # Clear the previous display and then write from the last message backwards
        for message in reversed(st.session_state.chat_history):
            if message["role"] == "assistant":
                st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
            else:
                st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
    else:
        st.error("Please refresh the page, submit and process a PDF file before asking questions.")


def explain_keyword(keyword, knowledge_level):
    if 'explanations' not in st.session_state:
        st.session_state.explanations = {}
    
    if keyword not in st.session_state.explanations:
        prompt_template = None
        if knowledge_level == "Beginner":
            prompt_template = PromptTemplate(
                input_variables=["keyword"],
                template=(
                "Explain the legal/financial term '{keyword}' in very simple English. "
                "Use words that are used by elementary students. The explanation should be for someone with a low to mid income (quintile 1-3) "
                "and primary to lower secondary education, around the level of 7 and 9 grader with low literacy and numeracy score"
                "Keep the sentence short and use one or two-syllable words."
                "Keep the explanation around 110 words. In the second paragraph add example in this format 'Example:' "
                )
            )
        elif knowledge_level == "Intermediate":
            prompt_template = PromptTemplate(
                input_variables=["keyword"],
                template=(
                "Explain the legal/financial term '{keyword}' with some technical details. "
                "The explanation should be for someone with a low to mid income (quintile 1-3) "
                "and upper secondary to post-secondary non-tertiary education (ISCED 3C - ISCED 4C). "
                "Assume medium literacy and numeracy scores. "
                "Use words with 1-3 syllables. "
                "Keep the sentences short and the explanation around 110 words. "
                "In the second paragraph, provide an example in this format 'Example:' "
                )
            )
        else:
            prompt_template = PromptTemplate(
                input_variables=["keyword"],
                template=(
                "Explain the legal/financial term '{keyword}' in detail, including technical aspects and implications. "
                "The explanation should be for someone with a mid to high income (quintile 4-5) "
                "and tertiary education (ISCED 5 and above). "
                "Assume high literacy and numeracy scores. "
                "Use precise and accurate terminology, but ensure clarity. "
                "Keep the explanation around 150 words. "
                "In the second paragraph, provide a comprehensive example in this format: 'Example:' "
                )
            )

        llm = ChatOpenAI(model="gpt-4")
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        response = llm_chain.run(keyword=keyword)
        response = response.replace("$", "\$")
        
        # Store the explanation in session state
        st.session_state.explanations[keyword] = response
    else:
        response = st.session_state.explanations[keyword]

    return response


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "extracted_legal_terms" not in st.session_state:
        st.session_state.extracted_legal_terms = None
    if "show_input" not in st.session_state:
        st.session_state.show_input = True  
    if "knowledge_level" not in st.session_state:
        st.session_state.knowledge_level = "Beginner"

    st.header("Chat with multiple PDFs :books:")

    if st.session_state.show_input:
        user_question = st.text_input("Ask a question about your documents:", key="user_question")
        if user_question:
            handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        
        knowledge_level = st.selectbox(
            "Select your knowledge level about legal documents:",
            ("Beginner", "Intermediate", "Advanced"),
            key="knowledge_level"
        )
        
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

               # Preprocess text and find matching topics
                preprocessed_text = preprocess_text(raw_text)
                matched_topics = find_matching_topics(preprocessed_text, unique_topics)
                
                # Store matched topics and raw text for conversation chain
                st.session_state.matched_topics = matched_topics
                st.session_state.raw_text = raw_text
        
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
        # User interface to show matched topics
        if "matched_topics" in st.session_state and st.session_state.matched_topics:
            st.subheader("Matched Topics (Top 5)")
            top_5_topics = st.session_state.matched_topics[:5]  # Get top 5 topics
            # Display topics with explanations
            for index, topic in enumerate(top_5_topics, start=1):
                st.write(f"{index}. {topic}")
                explanation = explain_keyword(topic, knowledge_level)
                st.write(f"Explanation: {explanation}")

if __name__ == '__main__':
    main()