import streamlit as st
import pandas as pd
import re
import emoji
import io
import openai
import time

# --------------------------------------------------------------------------
# 多言語キーワード辞書
# --------------------------------------------------------------------------
LANGUAGES = {
    'en': { # English
        'name': 'English',
        'POSITIVE_WORDS': ['good', 'great', 'excellent', 'fast', 'ok', 'love', 'happy', 'satisfied', 'delicious', 'tasty', 'fresh', 'recommended', 'perfect', 'nice', 'awesome'],
        'NEGATIVE_WORDS': ['bad', 'disappointed', 'slow', 'broken', 'damaged', 'wrong', 'not', 'terrible', 'awful', 'stale'],
        'PRODUCT_TOPICS_KEYWORDS': {
            'Taste': ['taste', 'flavor', 'delicious', 'tasty', 'sweet', 'sour'],
            'Quality': ['quality', 'fresh', 'condition', 'texture', 'genuine', 'original'],
            'Price': ['price', 'value', 'cheap', 'expensive', 'worth']
        }
    },
    'id': { # Indonesian
        'name': 'Indonesian',
        'POSITIVE_WORDS': ['bagus', 'sesuai', 'cepat', 'oke', 'baik', 'terima kasih', 'sukses', 'mantap', 'aman', 'rapi', 'murah', 'enak', 'lembut', 'puas', 'recommended', 'amanah', 'lengkap', 'bonus', 'suka', 'lezat', 'good', 'fresh', 'gift', 'alhamdulillah', 'realpict'],
        'NEGATIVE_WORDS': ['kecewa', 'ancur', 'hancur', 'lama', 'tidak', 'kurang', 'peok', 'rusak', 'masalah', 'jelek', 'ga', 'gak', 'tdk', 'retak', 'meleleh'],
        'PRODUCT_TOPICS_KEYWORDS': {
            'Rasa': ['rasa', 'enak', 'manis', 'coklat', 'lezat', 'gurih', 'strawberry'],
            'Kualitas': ['kualitas', 'bagus', 'sesuai', 'lengkap', 'produk', 'kondisi', 'isinya', 'fresh', 'tekstur', 'renyah'],
            'Harga': ['harga', 'murah', 'mahal', 'terjangkau', 'murcee']
        }
    },
    'ms': { # Malay
        'name': 'Malay',
        'POSITIVE_WORDS': ['baik', 'bagus', 'cepat', 'terbaik', 'sedap', 'puas hati', 'berbaloi', 'murah', 'terima kasih', 'selamat', 'kemas', 'original'],
        'NEGATIVE_WORDS': ['kecewa', 'lambat', 'rosak', 'pecah', 'salah', 'mahal', 'tak sedap', 'lembik'],
        'PRODUCT_TOPICS_KEYWORDS': {
            'Rasa': ['rasa', 'sedap', 'manis', 'masin', 'lazat'],
            'Kualiti': ['kualiti', 'keadaan', 'segar', 'tekstur', 'original'],
            'Harga': ['harga', 'murah', 'mahal', 'berbaloi']
        }
    },
    'vi': { # Vietnamese
        'name': 'Vietnamese',
        'POSITIVE_WORDS': ['tốt', 'nhanh', 'ngon', 'tuyệt vời', 'hài lòng', 'đúng', 'chất lượng', 'rẻ', 'cảm ơn', 'tươi'],
        'NEGATIVE_WORDS': ['tệ', 'thất vọng', 'chậm', 'hỏng', 'vỡ', 'sai', 'đắt', 'không ngon'],
        'PRODUCT_TOPICS_KEYWORDS': {
            'Hương vị': ['vị', 'ngon', 'ngọt', 'đậm đà'],
            'Chất lượng': ['chất lượng', 'tươi', 'kết cấu', 'nguyên bản'],
            'Giá cả': ['giá', 'rẻ', 'đắt', 'hợp lý']
        }
    },
    'th': { # Thai
        'name': 'Thai',
        'POSITIVE_WORDS': ['ดี', 'เร็ว', 'อร่อย', 'ยอดเยี่ยม', 'พอใจ', 'ถูกใจ', 'ขอบคุณ', 'คุ้มค่า', 'สดใหม่'],
        'NEGATIVE_WORDS': ['แย่', 'ผิดหวัง', 'ช้า', 'เสียหาย', 'แตก', 'แพง', 'ไม่อร่อย'],
        'PRODUCT_TOPICS_KEYWORDS': {
            'รสชาติ': ['รสชาติ', 'อร่อย', 'หวาน', 'กลมกล่อม'],
            'คุณภาพ': ['คุณภาพ', 'สด', 'สภาพ', 'เนื้อสัมผัส', 'ของแท้'],
            'ราคา': ['ราคา', 'ถูก', 'แพง', 'คุ้มค่า']
        }
    },
    'zh-cn': { # Simplified Chinese
        'name': 'Chinese (Simplified)',
        'POSITIVE_WORDS': ['好', '快', '好吃', '满意', '不错', '谢谢', '便宜', '新鲜', '推荐', '正品'],
        'NEGATIVE_WORDS': ['差', '失望', '慢', '坏了', '碎了', '贵', '不好吃', '假货'],
        'PRODUCT_TOPICS_KEYWORDS': {
            '味道': ['味道', '好吃', '口感', '甜', '咸'],
            '品质': ['品质', '质量', '新鲜', '正品', '状况'],
            '价格': ['价格', '便宜', '贵', '性价比']
        }
    }
}


# --------------------------------------------------------------------------
# AIによる自動判断機能
# --------------------------------------------------------------------------
def get_ai_decision(comment, api_key, model, language_name):
    try:
        openai.api_key = api_key
        prompt = f"""
        The following comment is in {language_name}.
        Judge if this comment is a useful review about the product itself (taste, quality, etc.).
        - Reviews about "packaging", "shipping", "seller service" are NOT useful.
        - Simple greetings or thanks are NOT useful.
        
        Comment: "{comment}"
        
        Based on these criteria, respond with only one word: KEEP or REMOVE.
        """
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3, temperature=0
        )
        return "KEEP" in response.choices[0].message.content.strip().upper()
    except Exception as e:
        st.warning(f"AI API Error: {e}. Keeping comment by default.")
        return True

# --------------------------------------------------------------------------
# データ読み込み関数
# --------------------------------------------------------------------------
def initial_process_excel(uploaded_file_object):
    try:
        df_full = pd.read_excel(uploaded_file_object, header=None, engine='openpyxl')
        brand_row, platform_row, df_body = df_full.iloc[0].ffill(), df_full.iloc[1], df_full.iloc[2:]
        records = [{'Brand': str(brand_row.iloc[c]).strip(), 'Platform': str(platform_row.iloc[c]).strip(), 'Original_Comment': str(comment)} for c in range(1, len(df_full.columns)) for comment in df_body[c].dropna()]
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"Excelファイルの処理中にエラー: {e}")
        return None

# --------------------------------------------------------------------------
# Streamlit アプリのUI部分
# --------------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title('🌍 多言語対応・高速AIレビュー分析ツール (v8.1)')

st.sidebar.header("⚙️ 設定")
api_key = st.sidebar.text_input("1. OpenAI APIキーを入力", type="password")

lang_display_names = {lang_data['name']: lang_code for lang_code, lang_data in LANGUAGES.items()}
selected_lang_name = st.sidebar.selectbox("2. コメントの言語を選択", options=lang_display_names.keys())
lang_code = lang_display_names[selected_lang_name]

model_name = st.sidebar.selectbox("3. AIモデルを選択", options=['gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo'])

if not api_key:
    st.warning("👈 サイドバーからOpenAI APIキーを入力してください。")
    st.stop()

uploaded_file = st.file_uploader("分析したいExcelファイルをアップロードしてください", type=['xlsx'])

if uploaded_file:
    if 'raw_df' not in st.session_state or st.session_state.get('uploaded_file_name') != uploaded_file.name:
        with st.spinner("ファイルを読み込んでいます..."):
            st.session_state.raw_df = initial_process_excel(uploaded_file)
            st.session_state.uploaded_file_name = uploaded_file.name
            if 'final_df' in st.session_state: del st.session_state.final_df

    raw_df = st.session_state.raw_df
    if raw_df is None: st.stop()
    st.success(f"{len(raw_df)}件の全コメントを読み込みました（言語: {selected_lang_name}）。")

    if st.button(f"🚀 {selected_lang_name}で高速ハイブリッド分析を実行", type="primary"):
        KW = LANGUAGES[lang_code]
        POSITIVE_WORDS, NEGATIVE_WORDS, PRODUCT_TOPICS_KEYWORDS = KW['POSITIVE_WORDS'], KW['NEGATIVE_WORDS'], KW['PRODUCT_TOPICS_KEYWORDS']
        
        with st.spinner("ステップ1: ルールベースで高速フィルタリング中..."):
            def clean_text(text): return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', emoji.demojize(text.lower()))).strip()
            raw_df['Cleaned_Comment'] = raw_df['Original_Comment'].apply(clean_text)
            
            reviews_to_keep, reviews_to_remove, reviews_for_ai = [], [], []
            product_keywords = [kw for kws in PRODUCT_TOPICS_KEYWORDS.values() for kw in kws]
            
            for idx, row in raw_df.iterrows():
                comment = row['Cleaned_Comment']
                if any(kw in comment for kw in product_keywords):
                    reviews_to_keep.append(idx)
                elif len(comment.split()) <= 3 and not any(w in comment for w in POSITIVE_WORDS + NEGATIVE_WORDS):
                    reviews_to_remove.append(idx)
                else:
                    reviews_for_ai.append(idx)
            st.info(f"AIによる精密検査対象は {len(reviews_for_ai)}件です。")

        if reviews_for_ai:
            progress_bar = st.progress(0, text="ステップ2: AIがグレーゾーンのコメントをレビュー中...")
            for i, idx in enumerate(reviews_for_ai):
                time.sleep(0.05)
                if get_ai_decision(raw_df.loc[idx, 'Original_Comment'], api_key, model_name, selected_lang_name):
                    reviews_to_keep.append(idx)
                progress_bar.progress((i + 1) / len(reviews_for_ai), text=f"AIレビュー中: {i+1}/{len(reviews_for_ai)}")
        
        final_df = raw_df.loc[reviews_to_keep].copy()
        st.success(f"分析完了！ {len(final_df)}件の有益な製品レビューが見つかりました。")

        with st.spinner("最終レポートを作成中..."):
            def analyze_sentiment(text):
                score = sum(1 for w in POSITIVE_WORDS if w in text) - sum(1 for w in NEGATIVE_WORDS if w in text)
                return 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral')
            
            def classify_topics(text):
                found = [topic for topic, kw_list in PRODUCT_TOPICS_KEYWORDS.items() if any(k in text for k in kw_list)]
                return ', '.join(found) if found else 'Lainnya (Other)'
            
            final_df['Sentiment'] = final_df['Cleaned_Comment'].apply(analyze_sentiment)
            final_df['Topics'] = final_df['Cleaned_Comment'].apply(classify_topics)
            
            summary_brand = final_df.groupby('Brand')['Sentiment'].value_counts().unstack(fill_value=0)
            summary_platform = final_df.groupby('Platform')['Sentiment'].value_counts().unstack(fill_value=0)
            
            for df in [summary_brand, summary_platform]:
                for sentiment in ['Positive', 'Negative', 'Neutral']: df[sentiment] = df.get(sentiment, 0)
                df['Total Comments'] = df[['Positive', 'Negative', 'Neutral']].sum(axis=1)

            st.session_state.final_df = final_df
            st.session_state.summary_brand_df = summary_brand
            st.session_state.summary_platform_df = summary_platform

    if 'final_df' in st.session_state:
        st.subheader("📊 分析結果")
        st.dataframe(st.session_state.final_df[['Brand', 'Platform', 'Original_Comment', 'Sentiment', 'Topics']])
        
        st.subheader("📋 サマリー")
        col1, col2 = st.columns(2)
        with col1:
            st.write("ブランド別サマリー")
            st.dataframe(st.session_state.summary_brand_df)
        with col2:
            st.write("プラットフォーム別サマリー")
            st.dataframe(st.session_state.summary_platform_df)

        # --- ★★★ ここからが復元されたExcelダウンロード機能 ★★★ ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            sheets_to_write = {
                '1. Product_Reviews_Data': st.session_state.final_df,
                '2. Summary_by_Brand': st.session_state.summary_brand_df,
                '3. Summary_by_Platform': st.session_state.summary_platform_df
            }
            workbook = writer.book
            header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#D7E4BC', 'border': 1})
            
            for sheet_name, df in sheets_to_write.items():
                is_summary = 'Summary' in sheet_name
                df.to_excel(writer, sheet_name=sheet_name, index=is_summary)
                worksheet = writer.sheets[sheet_name]

                # ヘッダーに書式を適用
                if is_summary:
                    worksheet.write(0, 0, df.index.name if df.index.name else 'Index', header_format)
                for col_num, value in enumerate(df.columns):
                    worksheet.write(0, col_num + (1 if is_summary else 0), value, header_format)
                
                # 列幅を自動調整
                all_cols = [df.index.to_series()] + [df[col] for col in df.columns] if is_summary else [df[col] for col in df.columns]
                all_names = [df.index.name or 'Index'] + df.columns.tolist() if is_summary else df.columns.tolist()
                for i, (col_data, col_name) in enumerate(zip(all_cols, all_names)):
                    # データとヘッダーの最大文字数を取得
                    max_len = max(col_data.astype(str).str.len().max(), len(str(col_name)))
                    worksheet.set_column(i, i, max_len + 2) # 少し余裕を持たせる
        
        st.download_button(
            label="📥 分析レポートをExcelでダウンロード",
            data=output.getvalue(),
            file_name=f"product_review_analysis_{uploaded_file.name}",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )