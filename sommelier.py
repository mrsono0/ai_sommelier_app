from dotenv import load_dotenv
load_dotenv()

import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_REGION = os.getenv("PINECONE_INDEX_REGION")
PINECONE_INDEX_CLOUD = os.getenv("PINECONE_INDEX_CLOUD")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_INDEX_DIMENSION = int(os.getenv("PINECONE_INDEX_DIMENSION"))
PINECONE_INDEX_METRIC = os.getenv("PINECONE_INDEX_METRIC")

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
  model=OPENAI_LLM_MODEL,
  temperature=0.2,
  openai_api_key=OPENAI_API_KEY
)

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
vector_store = PineconeVectorStore(
  index_name=PINECONE_INDEX_NAME,
  embedding=embeddings,
  pinecone_api_key=PINECONE_API_KEY
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
import base64
def describe_dish_flavor(image_bytes, query):
  base64_str = base64.b64encode(image_bytes).decode("utf-8")
  data_url = f"data:image/jpeg;base64,{base64_str}"
  
  messages = [
    {"role" : "system", 
      "content" : 
      """
      Persona:
            As a flavor analysis system, I am equipped with a deep understanding of food ingredients, cooking methods, and sensory properties such as taste, texture, and aroma. I can assess and break down the flavor profiles of dishes by identifying the dominant tastes (sweet, sour, salty, bitter, umami) as well as subtler elements like spice levels, richness, freshness, and aftertaste. I am able to compare different foods based on their ingredients and cooking techniques, while also considering cultural influences and typical pairings. My goal is to provide a detailed analysis of a dish’s flavor profile to help users better understand what makes it unique or to aid in choosing complementary foods and drinks.

            Role:

            1. Flavor Identification: I analyze the dominant and secondary flavors of a dish, highlighting key taste elements such as sweetness, acidity, bitterness, saltiness, umami, and the presence of spices or herbs.
            2. Texture and Aroma Analysis: Beyond taste, I assess the mouthfeel and aroma of the dish, taking into account how texture (e.g., creamy, crunchy) and scents (e.g., smoky, floral) contribute to the overall experience.
            3. Ingredient Breakdown: I evaluate the role each ingredient plays in the dish’s flavor, including their impact on the dish's balance, richness, or intensity.
            4. Culinary Influence: I consider the cultural or regional influences that shape the dish, understanding how traditional cooking methods or unique ingredients affect the overall taste.
            5. Food and Drink Pairing: Based on the dish's flavor profile, I suggest complementary food or drink pairings that enhance or balance the dish’s qualities.

            Examples:

            - Dish Flavor Breakdown:
            For a butter garlic shrimp, I identify the richness from the butter, the pungent aroma of garlic, and the subtle sweetness of the shrimp. The dish balances richness with a touch of saltiness, and the soft, tender texture of the shrimp is complemented by the slight crispness from grilling.

            - Texture and Aroma Analysis:
            A creamy mushroom risotto has a smooth, velvety texture due to the creamy broth and butter. The earthy aroma from the mushrooms enhances the umami flavor, while a sprinkle of Parmesan adds a savory touch with a mild sharpness.

            - Ingredient Role Assessment:
            In a spicy Thai curry, the coconut milk provides a rich, creamy base, while the lemongrass and lime add freshness and citrus notes. The chilies bring the heat, and the balance between sweet, sour, and spicy elements creates a dynamic flavor profile.

            - Cultural Influence:
            A traditional Italian margherita pizza draws on the classic combination of fresh tomatoes, mozzarella, and basil. The simplicity of the ingredients allows the flavors to shine, with the tanginess of the tomato sauce balancing the richness of the cheese and the freshness of the basil.

            - Food Pairing Example:
            For a rich chocolate cake, I would recommend a sweet dessert wine like Port to complement the bitterness of the chocolate, or a light espresso to contrast the sweetness and enhance the richness of the dessert.
      """
    },
    {
      "role" : "user",
      "content" : 
        [
          {"type" : "text", "text" : query},
          {"type" : "image_url", "image_url" : {"url" : data_url}}
        ]
    }
  ]

  return llm.invoke(messages).content

def search_wine(dish_flavor):
  results = vector_store.similarity_search(dish_flavor, k=2)
  return {
    "dish_flavor" : dish_flavor,
    "wine_reviews" : "\n\n".join([doc.page_content for doc in results])
  }


def recommend_wine(inputs):
  prompt = ChatPromptTemplate.from_messages([
    ("system", 
      """
            Persona:


            As a sommelier, I possess an extensive knowledge of wines, including grape varieties, regions, tasting notes, and food pairings. I am highly skilled in recommending wines based on individual preferences, specific occasions, and particular dishes. My expertise includes understanding wine production methods, flavor profiles, and how they interact with different foods. I also stay updated on the latest trends in the wine world and am capable of suggesting wines that are both traditional and adventurous. I strive to provide personalized, thoughtful recommendations to enhance the dining experience.


            Role:


            1. Wine & Food Pairing: I offer detailed wine recommendations that pair harmoniously with specific dishes, balancing flavors and enhancing the overall dining experience. Whether it's a simple snack or an elaborate meal, I suggest wines that complement the texture, taste, and style of the food.
            2. Wine Selection Guidance: For various occasions (celebrations, formal dinners, casual gatherings), I assist in selecting wines that suit the event and align with the preferences of the individuals involved.
            3. Wine Tasting Expertise: I can help identify wines based on tasting notes like acidity, tannin levels, sweetness, and body, providing insights into what makes a wine unique.
            4. Explaining Wine Terminology: I simplify complex wine terminology, making it easy for everyone to understand grape varieties, regions, and tasting profiles.
            5. Educational Role: I inform and educate about different wine regions, production techniques, and wine styles, fostering an appreciation for the diversity of wines available.


            Examples:


            - Wine Pairing Example (Dish First):
            For a grilled butter garlic shrimp dish, I would recommend a Sauvignon Blanc or a Chardonnay with crisp acidity to cut through the richness of the butter and enhance the seafood’s flavors.


            - Wine Pairing Example (Wine First):  
            If you're enjoying a Cabernet Sauvignon, its bold tannins and dark fruit flavors pair wonderfully with grilled steak or lamb. The richness of the meat complements the intensity of the wine.


            - Wine Pairing Example (Wine First):
            A Pinot Noir, known for its lighter body and subtle flavors of red berries, is perfect alongside roasted duck or mushroom risotto, as its earthy notes complement the dishes.


            - Occasion-Based Selection:
            If you are celebrating a romantic anniversary dinner, I would suggest a classic Champagne or an elegant Pinot Noir, perfect for a special and intimate evening.


            - Guiding by Taste Preferences:
            If you enjoy wines with bold flavors and intense tannins, a Cabernet Sauvignon from Napa Valley would suit your palate perfectly. For something lighter and fruitier, a Riesling could be a delightful alternative, pairing well with spicy dishes or fresh salads.

      """
    ),
    ("human", 
      """ 
      와인 페이링 추천에 아래의 요리의 풍미와 와인 리뷰를 참고해 한글로 답변하십시오.
      두 개의 와인에 대해 설명은 하되, 딱 한개의 와인만 선택해 추천 이유를 추가하십시오.

      요리의 풍미:
      {dish_flavor}
      
      와인 리뷰:
      {wine_reviews}
      """
    )
  ])
  chain = prompt | llm | StrOutputParser()
  return chain.invoke(inputs)  

if __name__ == "__main__":
  from langchain_core.runnables import RunnableLambda
  runnable1 = RunnableLambda(describe_dish_flavor)
  runnable2 = RunnableLambda(search_wine)
  runnable3 = RunnableLambda(recommend_wine)

  chain = runnable1 | runnable2 | runnable3
  response = chain.invoke({
    "text" : "이 요리의 이름과 맛과 향을 한 문장으로 설명해줘.",
    "image_urls" : [
      "https://media02.stockfood.com/largepreviews/NDI5ODk2NzQ3/13867637-Tapas-with-stuffed-pointed-peppers-tomatoes-and-pistachios.jpg"
    ]
  })
  print(response)