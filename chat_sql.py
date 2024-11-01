
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# from langchain.sql_database import SQLDatabase
from langchain_community.utilities import SQLDatabase
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class SQLChatBot:
    def __init__(self, db_uri):
        # Initialize tokenizer and model for the LLM
        self.tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
        self.model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b")
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)
        
        # Initialize the SQL database
        self.db = SQLDatabase.from_uri(db_uri)
        
        # Define a prompt template for SQL queries
        self.prompt_template = PromptTemplate(
            input_variables=["query"],
            template="Given the database context, answer the following SQL query: {query}"
        )
        
        # Set up the LLM chain for generating SQL responses
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def answer_query(self, query):
        # Execute the query using the LLM chain
        db_context = self.db.get_table_info()  # Retrieve context from SQL database
        full_query = f"{db_context} Query: {query}"
        return self.chain.run(query=full_query)


# class SQLChatBot:
#     def __init__(self):
#         # Initialize the LLM for text to SQL
#         self.tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
#         self.model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b")
#         self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
#         self.llm = HuggingFacePipeline(pipeline=self.pipeline)

#         # Set up database connection
#         self.engine = db.create_engine('sqlite:///your_database.db')


#     def answer_query(self, query):
#         # Convert natural language query to SQL
#         prompt = SQLPrompt(query)
#         sql_query = self.llm(prompt)

#         # Run SQL query and return results
#         with self.engine.connect() as connection:
#             result = connection.execute(sql_query)
#             df = pd.DataFrame(result.fetchall(), columns=result.keys())
#         return df.to_markdown()