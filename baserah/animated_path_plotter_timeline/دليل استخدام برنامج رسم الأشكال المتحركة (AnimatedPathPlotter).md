# دليل استخدام برنامج رسم الأشكال المتحركة (AnimatedPathPlotter)

## مقدمة

برنامج AnimatedPathPlotter هو أداة متطورة لإنشاء رسوم متحركة (أنيميشن) باستخدام لغة أوامر بسيطة. يمكنك استخدام هذا البرنامج لإنشاء أفلام كرتون، رسوم متحركة تفاعلية، وعروض تقديمية متحركة.

البرنامج يتيح لك:
- رسم أشكال متنوعة (مستطيلات، دوائر، مسارات مخصصة)
- تحريك الأشكال (الموقع، الدوران، التكبير/التصغير)
- تغيير الألوان والشفافية بشكل متحرك
- التحكم في الإطارات الرئيسية (keyframes) ومنحنيات التوقيت (easing)
- إنشاء شخصيات متحركة وتحريكها
- حفظ النتائج كملفات GIF أو MP4

## المكونات الرئيسية

البرنامج يتكون من ملفين رئيسيين:

1. **animated_path_plotter_timeline.py**: المكتبة الرئيسية التي تحتوي على كل وظائف الرسم والتحريك.
2. **animated_path_plotter_examples.py**: مجموعة من الأمثلة التوضيحية لاستخدام البرنامج.

## كيفية الاستخدام

### الاستخدام الأساسي

```python
from animated_path_plotter_timeline import AnimatedPathPlotter

# إنشاء كائن الراسم
plotter = AnimatedPathPlotter()

# تعريف أوامر الرسم والتحريك
commands = """
# إعداد التحريك
SET_ANIMATION_DURATION(5)
SET_ANIMATION_FPS(30)

# تعريف كائن
DEFINE_OBJECT("ball")
BEGIN_OBJECT("ball")
    PEN_DOWN()
    SET_COLOR_HEX("#FF0000")
    ENABLE_FILL()
    SET_FILL_COLOR_HEX("#FF0000")
    CIRCLE(0, 0, 20)
    PEN_UP()
END_OBJECT()

# تحريك الكائن
KEYFRAME_POSITION("ball", 0, 50, 50)
KEYFRAME_POSITION("ball", 5, 250, 50)
"""

# تنفيذ الأوامر
plotter.parse_and_execute(commands)

# عرض التحريك
plotter.plot(title="مثال بسيط", animate=True)
```

### تشغيل الأمثلة

يمكنك تشغيل الأمثلة المضمنة مباشرة:

```python
from animated_path_plotter_examples import example_bouncing_ball

# تشغيل مثال الكرة المتحركة
example_bouncing_ball()

# أو تشغيل جميع الأمثلة
from animated_path_plotter_examples import run_all_examples
run_all_examples()
```

## لغة الأوامر

### أوامر الرسم الأساسية

- `PEN_DOWN()`: بدء الرسم
- `PEN_UP()`: إيقاف الرسم
- `SET_COLOR_HEX("#RRGGBB")`: تعيين لون الخط
- `SET_THICKNESS(2)`: تعيين سماكة الخط
- `SET_OPACITY(0.5)`: تعيين الشفافية
- `ENABLE_FILL()`: تفعيل تعبئة الشكل
- `DISABLE_FILL()`: إلغاء تعبئة الشكل
- `SET_FILL_COLOR_HEX("#RRGGBB")`: تعيين لون التعبئة
- `MOVE_TO(x, y)`: الانتقال إلى نقطة
- `LINE_TO(x, y)`: رسم خط إلى نقطة
- `CURVE_TO(c1x, c1y, c2x, c2y, ex, ey)`: رسم منحنى بيزيه
- `CLOSE_PATH()`: إغلاق المسار
- `RECTANGLE(x, y, width, height)`: رسم مستطيل
- `CIRCLE(cx, cy, radius)`: رسم دائرة
- `VARIABLE_LINE_TO(x, y, thickness)`: رسم خط متغير السماكة

### أوامر التحريك

- `SET_ANIMATION_DURATION(seconds)`: تعيين مدة التحريك
- `SET_ANIMATION_FPS(frames)`: تعيين معدل الإطارات
- `SET_ANIMATION_OUTPUT("filename.gif")`: تعيين ملف الإخراج

### أوامر الكائنات

- `DEFINE_OBJECT("name")`: تعريف كائن جديد
- `BEGIN_OBJECT("name")`: بدء تعريف كائن
- `END_OBJECT()`: إنهاء تعريف كائن
- `DEFINE_GROUP("name")`: تعريف مجموعة كائنات
- `ADD_TO_GROUP("object", "group")`: إضافة كائن إلى مجموعة
- `SET_PARENT("child", "parent")`: تعيين علاقة أبوية بين كائنين

### أوامر الإطارات الرئيسية (Keyframes)

- `KEYFRAME_POSITION("object", time, x, y)`: تعيين موقع في وقت محدد
- `KEYFRAME_SCALE("object", time, scale)`: تعيين حجم في وقت محدد
- `KEYFRAME_ROTATION("object", time, angle)`: تعيين دوران في وقت محدد
- `KEYFRAME_COLOR("object", time, "#RRGGBB")`: تعيين لون خط في وقت محدد
- `KEYFRAME_FILL_COLOR("object", time, "#RRGGBB")`: تعيين لون تعبئة في وقت محدد
- `KEYFRAME_OPACITY("object", time, value)`: تعيين شفافية في وقت محدد
- `KEYFRAME_THICKNESS("object", time, value)`: تعيين سماكة خط في وقت محدد
- `KEYFRAME_FILL_ENABLED("object", time, true/false)`: تفعيل/إلغاء التعبئة في وقت محدد

### أوامر منحنيات التوقيت (Easing)

- `SET_EASING("object", "property", "easing_type")`: تعيين نوع منحنى التوقيت

أنواع منحنيات التوقيت المتاحة:
- `LINEAR`: خطي
- `EASE_IN`: تسارع تدريجي
- `EASE_OUT`: تباطؤ تدريجي
- `EASE_IN_OUT`: تسارع ثم تباطؤ
- `BOUNCE`: ارتداد
- `ELASTIC`: مرن
- `BACK`: رجوع
- `SINE`: جيبي

### أوامر التحكم

- `SHOW_TIMELINE_CONTROLS(true/false)`: إظهار/إخفاء عناصر التحكم في الخط الزمني
- `AUTO_PLAY(true/false)`: تشغيل تلقائي للتحريك
- `LOOP_ANIMATION(true/false)`: تكرار التحريك
- `SAVE_ANIMATION("filename.gif")`: حفظ التحريك كملف

## أمثلة متقدمة

### مثال 1: كرة متحركة مع تغيير اللون

```python
commands = """
SET_ANIMATION_DURATION(5)
SET_ANIMATION_FPS(30)

DEFINE_OBJECT("ball")
BEGIN_OBJECT("ball")
    PEN_DOWN()
    SET_COLOR_HEX("#000000")
    ENABLE_FILL()
    SET_FILL_COLOR_HEX("#FF0000")
    CIRCLE(0, 0, 20)
    PEN_UP()
END_OBJECT()

KEYFRAME_POSITION("ball", 0, 50, 50)
KEYFRAME_POSITION("ball", 2.5, 250, 250)
KEYFRAME_POSITION("ball", 5, 450, 50)

KEYFRAME_FILL_COLOR("ball", 0, "#FF0000")
KEYFRAME_FILL_COLOR("ball", 2.5, "#00FF00")
KEYFRAME_FILL_COLOR("ball", 5, "#0000FF")

SET_EASING("ball", "position", "bounce")
SET_EASING("ball", "fill_color", "linear")
"""
```

### مثال 2: شخصية كرتونية بسيطة

```python
commands = """
SET_ANIMATION_DURATION(6)
SET_ANIMATION_FPS(30)

# تعريف الرأس
DEFINE_OBJECT("head")
BEGIN_OBJECT("head")
    PEN_DOWN()
    SET_COLOR_HEX("#000000")
    ENABLE_FILL()
    SET_FILL_COLOR_HEX("#FFD700")
    CIRCLE(0, 0, 30)
    
    # العينان
    SET_FILL_COLOR_HEX("#FFFFFF")
    CIRCLE(-10, 10, 8)
    CIRCLE(10, 10, 8)
    
    # الفم
    DISABLE_FILL()
    MOVE_TO(-15, -10)
    CURVE_TO(-5, -20, 5, -20, 15, -10)
    PEN_UP()
END_OBJECT()

# تعريف الجسم
DEFINE_OBJECT("body")
BEGIN_OBJECT("body")
    PEN_DOWN()
    SET_COLOR_HEX("#000000")
    ENABLE_FILL()
    SET_FILL_COLOR_HEX("#FF6347")
    RECTANGLE(-25, 0, 50, 60)
    PEN_UP()
END_OBJECT()

# تعيين العلاقة الأبوية
SET_PARENT("head", "body")

# تحريك الشخصية
KEYFRAME_POSITION("body", 0, 100, 200)
KEYFRAME_POSITION("body", 3, 400, 200)
KEYFRAME_POSITION("body", 6, 100, 200)

KEYFRAME_POSITION("head", 0, 100, 140)
KEYFRAME_POSITION("head", 3, 400, 140)
KEYFRAME_POSITION("head", 6, 100, 140)
"""
```

## نصائح متقدمة

1. **استخدام العلاقات الأبوية**: يمكنك إنشاء تسلسل هرمي للكائنات باستخدام `SET_PARENT`. عند تحريك الكائن الأب، ستتحرك جميع الكائنات الابنة معه.

2. **منحنيات التوقيت**: استخدم منحنيات التوقيت المختلفة لإضفاء طابع واقعي على الحركة:
   - `BOUNCE`: مناسب للكرات والقفزات
   - `ELASTIC`: مناسب للحركات المرنة
   - `EASE_IN_OUT`: مناسب للحركات الطبيعية

3. **تحريك الألوان**: يمكنك تحريك الألوان لإنشاء تأثيرات مثل الوميض، التلاشي، وتغيير المزاج.

4. **تحريك الشفافية**: استخدم `KEYFRAME_OPACITY` لإنشاء تأثيرات الظهور والاختفاء.

5. **تحريك الدوران**: استخدم `KEYFRAME_ROTATION` لإنشاء حركات دورانية.

6. **تحريك الحجم**: استخدم `KEYFRAME_SCALE` لتكبير وتصغير الكائنات.

## الخاتمة

برنامج AnimatedPathPlotter يوفر لك أدوات قوية لإنشاء رسوم متحركة احترافية. يمكنك استخدامه لإنشاء:
- أفلام كرتون قصيرة
- رسوم متحركة تفاعلية
- عروض تقديمية متحركة
- شعارات متحركة
- تأثيرات بصرية

استمتع بإنشاء رسوماتك المتحركة!
