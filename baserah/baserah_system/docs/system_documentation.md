# توثيق نظام بصيرة

## مقدمة

نظام بصيرة هو نظام ذكاء اصطناعي مبتكر يجمع بين النواة الرياضياتية (معادلة الشكل العام)، والتمثيل الدلالي، والمعالجة الرمزية، والتعلم العميق والمعزز، لإنشاء نموذج لغوي معرفي توليدي متكامل.

النظام مصمم ليكون قادراً على:
- فهم وتوليد اللغة بناءً على الدلالات العميقة للحروف والكلمات
- استخلاص المعرفة من النصوص وتوليد معرفة جديدة
- التعلم والتطور ذاتياً من خلال آليات التعلم المتنوعة والتكيف المستمر
- دمج المعالجة الرمزية مع التعلم العميق والمعزز بطريقة مبتكرة

## المعمارية العامة

نظام بصيرة يتكون من عدة مكونات رئيسية متكاملة:

### 1. النواة الرياضياتية المعززة

النواة الرياضياتية المعززة هي أساس النظام وتتكون من خمسة مكونات رئيسية:

- **معادلة الشكل العام**: إطار عمل موحد لتمثيل الأشكال والأنماط وتحولاتها رياضياً
- **تكامل التعلم العميق والمعزز**: محولات معززة للتعلم العميق والتعلم المعزز
- **تفاعل الخبير/المستكشف**: نظام متقدم للتفاعل بين الخبير والمستكشف
- **تكامل الدلالات**: إدارة قاعدة البيانات الدلالية مع استخراج الخصائص والمحاور الدلالية
- **التحقق من النظام**: اختبارات شاملة للتحقق من تكامل وأداء جميع مكونات النظام

#### الفئات الرئيسية

- `GeneralShapeEquation`: الفئة الرئيسية لمعادلة الشكل العام
- `EquationType`: تعداد لأنواع المعادلات (شكل، نمط، سلوك، تحويل، قيد، مركب)
- `LearningMode`: تعداد لأنماط التعلم (بدون، موجه، معزز، غير موجه، هجين)
- `SymbolicExpression`: فئة للتعبيرات الرمزية
- `EquationMetadata`: فئة لبيانات وصفية للمعادلات

#### مثال استخدام

```python
from mathematical_core.general_shape_equation import GeneralShapeEquation, EquationType, LearningMode

# إنشاء معادلة شكل عام
equation = GeneralShapeEquation(
    equation_type=EquationType.SHAPE,
    learning_mode=LearningMode.HYBRID
)

# إضافة مكونات
equation.add_component("circle", "(x-cx)^2 + (y-cy)^2 - r^2")
equation.add_component("cx", "0")
equation.add_component("cy", "0")
equation.add_component("r", "5")

# تقييم المعادلة
result = equation.evaluate({"x": 0, "y": 0})
print(result)
```

### 2. المعمارية المعرفية اللغوية

المعمارية المعرفية اللغوية تحدد الإطار العام للنظام وتتكون من ثماني طبقات رئيسية:

- **النواة الرياضياتية** - تنفيذ معادلة الشكل العام ومحركات المعادلات
- **الأساس الدلالي** - تمثيل الدلالات للحروف والكلمات والمفاهيم
- **المعالجة الرمزية** - معالجة الرموز والمفاهيم والعلاقات
- **التمثيل المعرفي** - تمثيل المعرفة في شبكة مفاهيمية
- **التوليد اللغوي** - توليد اللغة بناءً على المعرفة والدلالات
- **استخلاص المعرفة** - استخلاص المعرفة من النصوص والبيانات
- **التطور الذاتي** - آليات التعلم والتكيف المستمر
- **طبقة التكامل** - تكامل جميع الطبقات السابقة

#### الفئات الرئيسية

- `CognitiveLinguisticArchitecture`: الفئة الرئيسية للمعمارية المعرفية اللغوية
- `ArchitecturalLayer`: تعداد لطبقات المعمارية
- `ProcessingMode`: تعداد لأنماط المعالجة
- `KnowledgeType`: تعداد لأنواع المعرفة
- `ArchitecturalComponent`: فئة لمكونات المعمارية

#### مثال استخدام

```python
from cognitive_linguistic.cognitive_linguistic_architecture import CognitiveLinguisticArchitecture

# إنشاء معمارية معرفية لغوية
architecture = CognitiveLinguisticArchitecture()

# طباعة مكونات المعمارية
for layer, components in architecture.layers.items():
    print(f"طبقة {layer.value}: {len(components)} مكون")
    for comp_name in components:
        component = architecture.get_component(comp_name)
        print(f"  - {comp_name}: {component.description}")
```

### 3. معالجة اللغة العربية

مكون معالجة اللغة العربية يوفر أدوات متقدمة لمعالجة اللغة العربية، بما في ذلك:

- **استخراج الجذور**: استخراج جذور الكلمات العربية
- **تحليل النحو**: تحليل البنية النحوية للجمل العربية
- **تحليل البلاغة**: تحليل البنية البلاغية والسمات الأسلوبية للنص العربي

#### الفئات الرئيسية

- `ArabicRootExtractor`: فئة لاستخراج جذور الكلمات العربية
- `ArabicSyntaxAnalyzer`: فئة لتحليل البنية النحوية للجمل العربية
- `ArabicRhetoricAnalyzer`: فئة لتحليل البنية البلاغية للنص العربي

#### مثال استخدام

```python
from arabic_nlp.morphology.root_extractor import ArabicRootExtractor
from arabic_nlp.syntax.syntax_analyzer import ArabicSyntaxAnalyzer
from arabic_nlp.rhetoric.rhetoric_analyzer import ArabicRhetoricAnalyzer

# إنشاء مستخرج الجذور
root_extractor = ArabicRootExtractor()

# استخراج جذور من نص
text = "العلم نور يضيء طريق الحياة"
roots = root_extractor.extract_roots(text)
for word, root in roots:
    print(f"{word}: {root}")

# تحليل النحو
syntax_analyzer = ArabicSyntaxAnalyzer()
syntax_analyses = syntax_analyzer.analyze(text)

# تحليل البلاغة
rhetoric_analyzer = ArabicRhetoricAnalyzer()
rhetoric_analysis = rhetoric_analyzer.analyze(text)
```

### 4. التوليد الإبداعي

مكون التوليد الإبداعي يوفر أدوات لتوليد محتوى إبداعي، بما في ذلك:

- **توليد الصور**: توليد صور بناءً على وصف نصي أو مفهوم دلالي أو معادلة رياضية
- **الخط العربي**: توليد صور للخط العربي
- **نقل الأسلوب**: تطبيق أسلوب صورة على صورة أخرى

#### الفئات الرئيسية

- `ImageGenerator`: فئة لتوليد الصور
- `GenerationMode`: تعداد لأنماط التوليد
- `GenerationParameters`: فئة لمعلمات التوليد
- `GenerationResult`: فئة لنتائج التوليد

#### مثال استخدام

```python
from creative_generation.image.image_generator import ImageGenerator, GenerationParameters, GenerationMode

# إنشاء مولد الصور
generator = ImageGenerator()

# توليد صورة من نص عربي
parameters = GenerationParameters(
    mode=GenerationMode.ARABIC_TEXT_TO_IMAGE,
    width=512,
    height=512,
    seed=42
)
result = generator.generate_image("العلم نور", parameters)

# حفظ الصورة
generator.save_image(result, "output.png")
```

### 5. تنفيذ الأكواد

مكون تنفيذ الأكواد يوفر بيئة آمنة لتنفيذ الأكواد بلغات برمجة مختلفة:

- **تنفيذ الأكواد**: تنفيذ الأكواد بلغات برمجة مختلفة
- **قيود الموارد**: تحديد قيود على الموارد المستخدمة
- **قيود الأمان**: تحديد قيود أمنية على الأكواد المنفذة

#### الفئات الرئيسية

- `CodeExecutor`: فئة لتنفيذ الأكواد
- `ProgrammingLanguage`: تعداد للغات البرمجة المدعومة
- `ExecutionConfig`: فئة لتكوين التنفيذ
- `ExecutionResult`: فئة لنتائج التنفيذ

#### مثال استخدام

```python
from code_execution.code_executor import CodeExecutor, ProgrammingLanguage, ExecutionConfig

# إنشاء منفذ الأكواد
executor = CodeExecutor()

# تنفيذ كود بايثون
code = """
print("Hello, world!")
for i in range(5):
    print(f"Number: {i}")
"""
result = executor.execute(code, ProgrammingLanguage.PYTHON)

# طباعة النتائج
print(f"Exit Code: {result.exit_code}")
print(f"Execution Time: {result.execution_time:.2f} seconds")
print("Standard Output:")
print(result.stdout)
print("Standard Error:")
print(result.stderr)
```

### 6. واجهة الويب

واجهة الويب توفر واجهة سهلة الاستخدام للتفاعل مع نظام بصيرة:

- **تحليل النص العربي**: تحليل النص العربي واستخراج الجذور وتحليل النحو والبلاغة
- **توليد الصور**: توليد صور بناءً على وصف نصي أو مفهوم دلالي أو معادلة رياضية
- **تنفيذ الأكواد**: تنفيذ الأكواد بلغات برمجة مختلفة
- **تقييم المعادلات**: تقييم المعادلات الرياضية

#### مثال استخدام

```bash
# تشغيل واجهة الويب
python baserah_system/interfaces/web/app.py
```

ثم افتح المتصفح على العنوان `http://localhost:5000` للوصول إلى واجهة الويب.

## تشغيل النظام

### متطلبات النظام

- Python 3.8+
- PyTorch 1.9+
- NumPy
- SymPy
- Matplotlib
- Pandas
- NetworkX
- Flask (للواجهة الويب)
- PIL (لمعالجة الصور)

### التثبيت

1. قم بنسخ المستودع:
```bash
git clone https://github.com/username/baserah_system.git
cd baserah_system
```

2. قم بإنشاء بيئة افتراضية وتفعيلها:
```bash
python -m venv venv
source venv/bin/activate  # على Linux/Mac
venv\Scripts\activate  # على Windows
```

3. قم بتثبيت المتطلبات:
```bash
pip install -r requirements.txt
```

### تشغيل النظام

لتشغيل النظام الأساسي:

```bash
python baserah_system/main.py
```

لتشغيل واجهة الويب:

```bash
python baserah_system/interfaces/web/app.py
```

لتشغيل الاختبارات:

```bash
python -m unittest discover baserah_system/tests
```

## التطوير المستقبلي

### المكونات المخطط لها

- **تحسين معالجة اللغة العربية**: تحسين دقة وأداء مكونات معالجة اللغة العربية
- **تحسين التوليد الإبداعي**: تحسين جودة الصور المولدة وإضافة دعم لتوليد الفيديو
- **تحسين تنفيذ الأكواد**: إضافة دعم للمزيد من لغات البرمجة وتحسين الأمان
- **تحسين واجهة المستخدم**: تحسين واجهة الويب وإضافة واجهة سطح المكتب
- **تحسين التوثيق**: توثيق شامل للنظام ومكوناته

### المساهمة

نرحب بالمساهمات! يرجى اتباع الخطوات التالية:

1. قم بعمل fork للمستودع
2. قم بإنشاء فرع جديد (`git checkout -b feature/amazing-feature`)
3. قم بإجراء التغييرات
4. قم بعمل commit للتغييرات (`git commit -m 'Add some amazing feature'`)
5. قم بدفع التغييرات إلى الفرع (`git push origin feature/amazing-feature`)
6. قم بفتح طلب سحب

## الترخيص

هذا المشروع مرخص تحت رخصة MIT - انظر ملف LICENSE للتفاصيل.

## الاتصال

لمزيد من المعلومات، يرجى التواصل مع فريق تطوير نظام بصيرة.

---

**ملاحظة**: هذا المشروع قيد التطوير المستمر، وقد تتغير المعمارية والمكونات مع تقدم المشروع.
