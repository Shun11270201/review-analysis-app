import pandas as pd
import re
import emoji

def clean_and_process_excel(file_path):
    """
    Excelファイルを読み込み、データクレンジングを行い、整形されたDataFrameを返す関数
    """
    try:
        # --- ステップ1: データの読み込み ---
        # 1行目（ヘッダー）を読み込み、ブランド名を取得
        df_header_brand = pd.read_excel(file_path, header=0, nrows=1)
        # 2行目（サブヘッダー）を読み込み、プラットフォーム名を取得
        df_header_platform = pd.read_excel(file_path, header=1, nrows=1)
        # 3行目以降の本体データを読み込む
        df_body = pd.read_excel(file_path, header=2)

    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。")
        return None
    except Exception as e:
        print(f"Excelファイルの読み込み中にエラーが発生しました: {e}")
        return None

    # --- ステップ2: ヘッダー情報の整形 ---
    # 1行目のブランド名はセルが結合されているため、空白を前の値で埋める (Forward Fill)
    df_header_brand.fillna(method='ffill', axis=1, inplace=True)
    
    # ブランドとプラットフォームを対応付ける辞書を作成
    # 'Unnamed: 0' のような列は除外する
    brands = df_header_brand.iloc[0].drop(df_header_brand.columns[0]).tolist()
    platforms = df_header_platform.iloc[0].drop(df_header_platform.columns[0]).tolist()
    
    # bodyの列名を 'Brand_Platform' の形に統一
    new_columns = ['Comments'] # 最初の列は 'Comments'
    for brand, platform in zip(brands, platforms):
        new_columns.append(f"{brand}_{platform}")
        
    # 'Comments'列をインデックスに設定してから列名を変更
    df_body = df_body.rename(columns={'Unnamed: 0': 'Comments'})
    df_body.columns = new_columns

    # --- ステップ3: データの構造化（縦長データへの変換） ---
    df_melted = df_body.melt(
        id_vars=['Comments'],
        var_name='Brand_Platform',
        value_name='Original_Comment'
    )

    # 'Brand_Platform'列を '_' で分割して、新しい列を作成
    df_melted[['Brand', 'Platform']] = df_melted['Brand_Platform'].str.split('_', expand=True)

    # 不要な列を削除し、順番を整理
    df_structured = df_melted.drop(columns=['Comments', 'Brand_Platform'])
    df_structured = df_structured[['Brand', 'Platform', 'Original_Comment']]

    # --- ステップ4: データクレンジング ---
    # 欠損値（空のセル）を削除
    df_cleaned = df_structured.dropna(subset=['Original_Comment'])
    df_cleaned = df_cleaned[df_cleaned['Original_Comment'] != 'No data']
    
    # コメントを文字列型に変換
    df_cleaned['Original_Comment'] = df_cleaned['Original_Comment'].astype(str)

    # クレンジング関数
    def clean_text(text):
        # emojiライブラリで絵文字をテキスト表現に変換し、それを削除
        text = emoji.demojize(text)
        text = re.sub(r':[a-zA-Z_]+:', '', text)
        # 記号や句読点を削除
        text = re.sub(r'[^\w\s]', '', text)
        # 連続する空白や改行を一つのスペースに置換し、前後の空白を削除
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # 'Cleaned_Comment'列を作成
    df_cleaned['Cleaned_Comment'] = df_cleaned['Original_Comment'].apply(clean_text)

    # クレンジング後に空になったり、意味をなさない短いコメントを削除
    # 例: 3文字以下のコメントは削除
    df_final = df_cleaned[df_cleaned['Cleaned_Comment'].str.len() > 3].copy()
    
    # インデックスをリセット
    df_final.reset_index(drop=True, inplace=True)
    
    return df_final

# --- メインの処理 ---
if __name__ == "__main__":
    # ここにExcelファイル名を入力してください
    input_excel_file = 'Lotte_CEP 発話分析.xlsx'
    
    # 関数を実行してクレンジング
    cleaned_df = clean_and_process_excel(input_excel_file)
    
    if cleaned_df is not None:
        # 結果をCSVファイルとして保存
        output_csv_file = 'cleaned_comments_output.csv'
        cleaned_df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
        
        print(f"データクレンジングが完了しました。")
        print(f"結果は '{output_csv_file}' に保存されました。")
        print("\n--- 処理後のデータ（先頭10件） ---")
        print(cleaned_df.head(10))