# portfolio-word-embedding

文章の分散表現モデルの学習と成果物のアップロード


---

## Features

- Download slack messages from slack DWH on BigQuery
- NLP preparation: Remove Noses, Morphological Analysis ... and so on.
- Build word embedding model with doc2vec.


## Setup

1. install dependencies
2. set environment values

### 1. Install dependencies

```bash
# if use venv
python -m venv venv
source ./venv/bin/activate

# in venv ----
pip install -U pip
pip install -r requirements.txt
```


### 2. Set environment values

```bash
cp .env-template .env
nano .env
```

```bash
# Set each values
BQ_PROJECT_ID=$YOUR_BIGQUERY_PROJECT_ID
GAE_DEFAULT_BUCKET_NAME=$YOUR_APPENGINE_DEFAULT_BUCKET_NAME
GOOGLE_APPLICATION_CREDENTIALS=$YOUR_GOOGLE_APPLICATION_CREDENTIALS_PATH
```

- `BQ_PROJECT_ID`
  - Google Cloud のプロジェクトIDを指定します
  - ソースデータが蓄積されているBigQueryのデータセットを特定するために利用します
- `GAE_DEFAULT_BUCKET_NAME`
  - 学習の成果物を保存するための Cloud Storage Bucket 名を記述します
  - 本サンプルでは、レコメンドAPIサーバーを起動するGoogle App Engineのデフォルトバケットを指定しています。（通常 `$PROJECT-ID.appspot.com` という形式になっています）
  - [ref: Google App Engineのデフォルトバケットとは？](https://cloud.google.com/appengine/quotas?hl=ja#Default_Gcs_Bucket)
- `GOOGLE_APPLICATION_CREDENTIALS`
  - サービスアカウントの秘匿情報を記述したファイルのパスを指定します。
  - [ref: サービスアカウント認証][gcp_auth_getstarted]



## Usage

スクリプトは以下の機能に分けられます。

各スクリプトを実行する前に、すべてのセットアップを完了させてください。

1. Make dataset
2. Train model
3. Vectorize conversations by users
4. Simulate matching locally
5. Upload model


### 1. Make dataset

BigQuery に蓄積されたデータを用いて学習用データセット、テスト用データセットを作成、保存します。


```bash
python make_dataset.py
```


### 2. Train model

前項で作成したデータセットを用いて、モデルを学習させ、学習済みモデルを保存します。

```bash
python train.py
```


### 3. Vectorize conversations by users

学習済みモデルを用いて、ユーザーごとの発言特性ベクトルを算出します。


```bash
python vectorize_user.py
```

### 4. Simulate matching locally

学習済みモデルを用いて、簡単なシミュレーションを行うことができます。

```bash
# matching simulation - with cmd args
python matching_simulator.py --cmd Pythonに詳しいのは誰？

# matching simulation - with a file
# $ cat question.txt
# Pythonに詳しいのは誰？
python matching_simulator.py --file question.txt

# matching simulation - with user id
python matching_simulator.py --user $USER_ID
```


### 5. Upload model

学習済みモデル、ユーザーごとの発言特性ベクトルデータを Cloud Storage にアップロードします。


```bash
python upload_model.py
```


---

## References

See:

- [Google Cloud Authentication Getting Started | Google Cloud Docs][gcp_auth_getstarted]
- [BQ Python client org][python_client_for_gbq]
- [pandas-gbq からの移行 | Google Cloud Docs][pandas_gbq_and_gbq]
- [Uploading objects with gsutil | Google Cloud Docs][gsutil_cp_to_gcs]
- [Uploading objects with python | Google Cloud Docs][upload_to_gcs_python]
- [gsutil cp | Google Cloud Reference][gsutil_cp_ref]
- [Creating buckets with gsutil | Google Cloud Docs][gsutil_mb]
- [Creating buckets with python | Google Cloud Docs][create_bucket_python]
- [GCS Storage classes | Google Cloud Docs][gcs_storage_classes]
- [spaCy Usage][spacy_usage]
- [spaCy Models][spacy_models]
- [spaCy API][spacy_api]
- [spaCy Universe][spacy_univ]
- [spaCy Course][spacy_course]

[spacy_usage]: https://spacy.io/usage
[spacy_models]: https://spacy.io/models
[spacy_api]: https://spacy.io/api
[spacy_univ]: https://spacy.io/universe
[spacy_course]: https://course.spacy.io/ja/
[python_client_for_gbq]: https://googleapis.dev/python/bigquery/latest/index.html
[pandas_gbq_and_gbq]: https://cloud.google.com/bigquery/docs/pandas-gbq-migration?hl=ja
[gsutil_cp_to_gcs]: https://cloud.google.com/storage/docs/uploading-objects?hl=ja#gsutil
[gsutil_cp_ref]: https://cloud.google.com/storage/docs/gsutil/commands/cp
[gsutil_mb]: https://cloud.google.com/storage/docs/creating-buckets?hl=ja
[gcs_storage_classes]: https://cloud.google.com/storage/docs/storage-classes
[upload_to_gcs_python]: https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python
[create_bucket_python]: https://cloud.google.com/storage/docs/creating-buckets#storage-create-bucket-code_samples
[gcp_auth_getstarted]: https://cloud.google.com/docs/authentication/getting-started
