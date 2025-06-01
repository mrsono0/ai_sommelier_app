# uv add streamlit watchdog

import base64
import streamlit as st
from sommelier import recommend_wine
from sommelier import describe_dish_flavor
from sommelier import search_wine

# streamlit run app.py

st.title("Sommelier AI")
col1, col2 = st.columns([3, 1])

with col1:
  upload_image = st.file_uploader("요리 이미지를 업로드하세요.", type=["jpg", "png", "jpeg"])
  user_prompt = st.text_input("프롬프트를  입력하세요", "이 요리에 어울리는 와인을 추천해주세요.")

with col2:
  if upload_image:
    st.image(upload_image, caption="업로드된 요리 이미지", use_container_width=True)

with col1:
  if st.button("추천 하기"):
    if not upload_image:
      st.error("이미지를 업로드하세요.")
    else:
      with st.spinner("1단계: 요리의 맛과 향을 분석하는 중..."):
        dish_flavor = describe_dish_flavor(upload_image.read(), "이 요리의 이름과 맛과 향을 한 문장으로 설명해줘.")
        # https://getemoji.com/
        st.markdown("### 🥙 요리의 맛과 향 분석 결과")
        st.info(f"{dish_flavor}")

      with st.spinner("2단계: 요리에 어울리는 와인 리뷰를 검색하는 중..."):
        wine_search_result = search_wine(dish_flavor)
        st.markdown("### 🍷 요리에 어울리는 와인 리뷰 검색 결과")
        st.info(f"{wine_search_result["wine_reviews"]}")

      with st.spinner("3단계: AI 소물리에가 와인 페어링에 대한 추천글을 생성하는 중..."):
        wine_recommendation = recommend_wine({
          "dish_flavor" : dish_flavor,
          "wine_reviews" : wine_search_result["wine_reviews"]
        })
        st.markdown("### 🏆 AI 소물리에의 와인 페어링 추천")
        st.info(f"{wine_recommendation}")

      st.success("🔚 와인 추천이 완료되었습니다.")
