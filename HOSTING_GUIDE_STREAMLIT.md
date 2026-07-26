# استضافة المشروع (موديلك انت — pipeline كامل) على Streamlit Community Cloud

## 1. جرّبه محلي الأول
```bash
cd streamlit_app_v2
python -m venv venv
source venv/bin/activate        # على Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
هيفتحلك على `http://localhost:8501`. جربت الملفين بتوعك فعلًا هنا واشتغلوا صح
(المثال الخبيث طلع باحتمال ~99%، والحميد ~3%).

## 2. ارفع المجلد على GitHub
```
streamlit_app_v2/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── artifacts/
    ├── best_breast_cancer_pipeline.pkl
    └── label_encoder.pkl
```

```bash
cd streamlit_app_v2
git init
git add .
git commit -m "Breast cancer diagnosis - my own stacking pipeline"
git branch -M main
git remote add origin https://github.com/USERNAME/breast-cancer-diagnosis.git
git push -u origin main
```

## 3. استضافة على Streamlit Community Cloud (مجاني)
1. [share.streamlit.io](https://share.streamlit.io) → سجّل دخول بحساب GitHub.
2. **New app** → اختار الـ repo والـ branch (`main`) والملف الرئيسي `app.py`
   (أو `streamlit_app_v2/app.py` لو مش في جذر الـ repo).
3. **Deploy** واستنى دقيقة/اتنين لحد ما يثبّت المكتبات.
4. هتاخد رابط زي `https://username-app-name.streamlit.app`.

## ملاحظات تخص موديلك تحديدًا

### أرقام الأداء (Accuracy / Recall / ROC-AUC)
المشروع دلوقتي بيشتغل من غير `model_summary.json`، فالواجهة بتعرض بدلها
اسم الموديل وعدد الـ features والـ threshold. لو حابب تظهر الأرقام الحقيقية
بتاعت موديلك فوق الصفحة زي الأول، ابعتلي:
- Accuracy على test set
- Recall للـ malignant
- ROC-AUC
- الـ threshold اللي استخدمته (لو ضبطته، وإلا سيبته 0.5 زي الافتراضي دلوقتي)

وأنا أعمللك `model_summary.json` وأربطه بالواجهة تلقائي.

### عدد الـ features
موديلك مدرّب على **24 feature بس** مش 30 — لاحظت إن `perimeter_*` و`area_*`
(المحيط والمساحة) مش موجودين في الـ pipeline، غالبًا اتشالوا عشان بيرتبطوا
ارتباط قوي جدًا بـ `radius_*` (نفس المعلومة تقريبًا). الفورم اتظبطت تلقائيًا
على الـ 24 دول بس.

### إصدارات المكتبات
`requirements.txt` مضبوط بالظبط على الإصدارات اللي جربتها ونجحت مع موديلك
(`scikit-learn==1.6.1`, `xgboost==3.3.0`, `lightgbm==4.5.0`) — متغيرهاش، لأن
موديلك اتدرب بإصدار `scikit-learn 1.6.1` بالظبط، وإصدارات تانية ممكن تدّي error
أو نتايج مختلفة عند التحميل.
