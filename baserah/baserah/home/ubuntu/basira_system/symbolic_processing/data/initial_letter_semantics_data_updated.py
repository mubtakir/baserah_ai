#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# قاعدة بيانات دلالات الحروف الأولية
# تم تحديثها بواسطة مانوس

initial_letter_semantics_data = {
    "ar": {  # الحروف العربية
        "ا": {
            "character": "ا",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "حنجري",
                "articulation_method": "مد",
                "sound_echoes": ["صوت المفاجأة (مع الهمزة)", "التعجب (مع الهمزة)"],
                "general_sound_quality": "صوت أساسي، مفتوح"
            },
            "visual_form_semantics": ["الاستقامة", "الارتفاع", "العلو", "البداية"],
            "core_semantic_axes": {
                "magnitude_elevation": ("عظمة/ارتفاع/علو (حسي ومعنوي)", "صغر/انخفاض")
            },
            "general_connotations": ["العظمة", "الارتفاع", "العلو (الحسي والمعنوي)"],
            "examples_from_basil": [
                "أ ل م (في سورة البقرة): الألف للعظمة والتعظيم."
            ],
            "notes_from_basil": "الألف يوحي للعظمة والارتفاع والعلو. الهمزة: صوت المفاجأة والرعب والصدمة وللتعجب. حروف العلة (ا, و, ي) مع الهمزة تفيد التعجب والفزع والخوف والفرح."
        },
        "ب": {
            "character": "ب",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "شفوي",
                "articulation_method": "انفجاري",
                "sound_echoes": ["صوت ارتطام", "صوت امتلاء"],
                "general_sound_quality": "صوت ارتطام وامتلاء وتشبع"
            },
            "visual_form_semantics": ["حوض أو إناء (الشكل القديم كمربع)", "بوابة (لأنه شفوي، مقدمة الفم)"],
            "core_semantic_axes": {
                "containment_transfer": ("امتلاء/تشبع/حمل/نقل", "إفراغ/ترك")
            },
            "general_connotations": ["الامتلاء", "التشبع", "النقل (سبب منطقي للامتلاء)"],
            "examples_from_basil": [
                "بحر/نهر: نقطة الباء السفلية توحي بقطرة ماء انصبت.",
                "طلب، حلب، سلب، نهب: فيها معنى الانتقال.",
                "بلع، بلغ، بعد، قرب: تفيد الانتقال والتشبع.",
                "اسم الحرف 'باء' من باء يبوء (امتلأ به وبان عليه)."
            ],
            "notes_from_basil": "الباء للامتلاء والتشبع والنقل. المعاني ترتبط ارتباط سببي ومنطقي. الحروف الشفوية كأنها حروف مادية ترسم الواقع العملياتي والحركي الملموس."
        },
        "ت": {
            "character": "ت",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "أسناني لثوي",
                "articulation_method": "انفجاري مهموس",
                "sound_echoes": ["صوت النقر الخفيف", "صوت الطرق الناعم"],
                "general_sound_quality": "صوت خفيف ناعم"
            },
            "visual_form_semantics": ["نقطتان فوق خط أفقي", "علامة صغيرة"],
            "core_semantic_axes": {
                "completion_continuation": ("إتمام/إكمال/تحقق", "بداية/استمرار")
            },
            "general_connotations": ["الإتمام", "التحقق", "الاكتمال", "التأنيث"],
            "examples_from_basil": [
                "تم، تمام، أتم: تفيد الإتمام والاكتمال",
                "تاب، توب: تفيد الرجوع والعودة إلى الأصل"
            ],
            "notes_from_basil": "التاء للإتمام والتحقق والاكتمال. وهي أيضاً علامة التأنيث في اللغة العربية. التاء المربوطة (ة) تفيد الاحتواء والإحاطة والتأنيث."
        },
        "ث": {
            "character": "ث",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "أسناني",
                "articulation_method": "احتكاكي مهموس",
                "sound_echoes": ["صوت النفث", "صوت الانتشار"],
                "general_sound_quality": "صوت انتشار وتفرق"
            },
            "visual_form_semantics": ["ثلاث نقاط فوق خط أفقي", "انتشار وتعدد"],
            "core_semantic_axes": {
                "dispersion_collection": ("انتشار/تفرق/تعدد", "تجمع/تركيز")
            },
            "general_connotations": ["الانتشار", "التفرق", "التعدد", "الكثرة"],
            "examples_from_basil": [
                "ثر، ثرى، ثروة: تفيد الكثرة والانتشار",
                "بث، نفث: تفيد النشر والتفريق"
            ],
            "notes_from_basil": "الثاء للانتشار والتفرق والتعدد. النقاط الثلاث فوق الحرف توحي بالتعدد والانتشار."
        },
        "ج": {
            "character": "ج",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "وسط الحنك",
                "articulation_method": "مركب",
                "sound_echoes": ["صوت التجمع", "صوت الاحتواء"],
                "general_sound_quality": "صوت تجمع واحتواء"
            },
            "visual_form_semantics": ["وعاء أو حوض", "تجويف"],
            "core_semantic_axes": {
                "containment_gathering": ("احتواء/تجمع/جمع", "تفرق/انتشار")
            },
            "general_connotations": ["الاحتواء", "التجمع", "الجمع", "التجويف"],
            "examples_from_basil": [
                "جمع، جماعة: تفيد التجمع والاجتماع",
                "جوف، جيب: تفيد التجويف والاحتواء",
                "جبل: تجمع الصخور والتراب في مكان مرتفع"
            ],
            "notes_from_basil": "الجيم للاحتواء والتجمع والتجويف. شكل الحرف يشبه الوعاء أو الحوض الذي يحتوي شيئاً ما."
        },
        "ح": {
            "character": "ح",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "حلقي",
                "articulation_method": "احتكاكي مهموس",
                "sound_echoes": ["صوت التنفس العميق", "صوت الحياة"],
                "general_sound_quality": "صوت عميق من الحلق"
            },
            "visual_form_semantics": ["دائرة مفتوحة", "حدود", "إحاطة"],
            "core_semantic_axes": {
                "life_boundary": ("حياة/حيوية/حركة", "موت/سكون"),
                "containment_limitation": ("إحاطة/حدود/حماية", "انفتاح/تجاوز")
            },
            "general_connotations": ["الحياة", "الحيوية", "الحركة", "الإحاطة", "الحدود", "الحماية"],
            "examples_from_basil": [
                "حي، حياة: تفيد الحياة والحيوية",
                "حوط، حاط: تفيد الإحاطة والحماية",
                "حد، حدود: تفيد التحديد والفصل"
            ],
            "notes_from_basil": "الحاء للحياة والحيوية والإحاطة. صوت الحاء يشبه صوت التنفس العميق الذي هو أساس الحياة. شكل الحرف يوحي بالإحاطة والحدود."
        },
        "خ": {
            "character": "خ",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "حلقي",
                "articulation_method": "احتكاكي مجهور",
                "sound_echoes": ["صوت الخروج", "صوت النفاذ"],
                "general_sound_quality": "صوت خشن من الحلق"
            },
            "visual_form_semantics": ["دائرة مفتوحة مع نقطة", "ثقب أو فتحة"],
            "core_semantic_axes": {
                "penetration_exit": ("خروج/نفاذ/اختراق", "دخول/بقاء")
            },
            "general_connotations": ["الخروج", "النفاذ", "الاختراق", "الفراغ"],
            "examples_from_basil": [
                "خرج، خروج: تفيد الخروج والانفصال",
                "خرق، اختراق: تفيد النفاذ والاختراق",
                "خلا، خلاء: تفيد الفراغ والخلو"
            ],
            "notes_from_basil": "الخاء للخروج والنفاذ والاختراق. النقطة فوق الحرف توحي بثقب أو فتحة للخروج."
        },
        "د": {
            "character": "د",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "أسناني لثوي",
                "articulation_method": "انفجاري مجهور",
                "sound_echoes": ["صوت الدق", "صوت الضرب"],
                "general_sound_quality": "صوت قوي حاد"
            },
            "visual_form_semantics": ["قوس مغلق", "باب"],
            "core_semantic_axes": {
                "entry_access": ("دخول/ولوج/وصول", "خروج/انفصال")
            },
            "general_connotations": ["الدخول", "الولوج", "الوصول", "الباب"],
            "examples_from_basil": [
                "دخل، دخول: تفيد الدخول والولوج",
                "درب، درج: تفيد المسار والطريق",
                "دار: تفيد المكان المحيط والمغلق"
            ],
            "notes_from_basil": "الدال للدخول والولوج والوصول. شكل الحرف يشبه الباب أو المدخل."
        },
        "ذ": {
            "character": "ذ",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "أسناني",
                "articulation_method": "احتكاكي مجهور",
                "sound_echoes": ["صوت الذوبان", "صوت الانتشار اللطيف"],
                "general_sound_quality": "صوت انسيابي لين"
            },
            "visual_form_semantics": ["خط مع نقطة", "إشارة"],
            "core_semantic_axes": {
                "indication_reference": ("إشارة/تذكير/ذكر", "نسيان/إهمال")
            },
            "general_connotations": ["الإشارة", "التذكير", "الذكر", "الانتشار اللطيف"],
            "examples_from_basil": [
                "ذكر، تذكير: تفيد الذكر والتذكير",
                "ذاب، ذوبان: تفيد الانتشار والتلاشي اللطيف",
                "ذهب: تفيد الانتقال والمضي"
            ],
            "notes_from_basil": "الذال للإشارة والتذكير والانتشار اللطيف. النقطة فوق الحرف توحي بالإشارة والتنبيه."
        },
        "ر": {
            "character": "ر",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "لثوي",
                "articulation_method": "تكراري",
                "sound_echoes": ["صوت التكرار", "صوت الحركة المستمرة"],
                "general_sound_quality": "صوت متكرر متحرك"
            },
            "visual_form_semantics": ["رأس منحني", "حركة دائرية"],
            "core_semantic_axes": {
                "repetition_movement": ("تكرار/حركة/استمرارية", "توقف/ثبات")
            },
            "general_connotations": ["التكرار", "الحركة", "الاستمرارية", "الدوران"],
            "examples_from_basil": [
                "كرر، تكرار: تفيد التكرار والإعادة",
                "دار، دوران: تفيد الحركة الدائرية",
                "جرى، جريان: تفيد الحركة المستمرة"
            ],
            "notes_from_basil": "الراء للتكرار والحركة والاستمرارية. صوت الراء متكرر بطبيعته، وشكل الحرف يوحي بالحركة الدائرية."
        },
        "ز": {
            "character": "ز",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "لثوي",
                "articulation_method": "احتكاكي مجهور",
                "sound_echoes": ["صوت الزحف", "صوت الانزلاق"],
                "general_sound_quality": "صوت انسيابي مستمر"
            },
            "visual_form_semantics": ["خط مستقيم مع نقطة", "مسار"],
            "core_semantic_axes": {
                "movement_progression": ("حركة/تقدم/زيادة", "ثبات/نقصان")
            },
            "general_connotations": ["الحركة", "التقدم", "الزيادة", "الانزلاق"],
            "examples_from_basil": [
                "زاد، زيادة: تفيد النمو والزيادة",
                "زحف، انزلاق: تفيد الحركة الانسيابية",
                "زمن، زمان: تفيد الاستمرارية والتقدم"
            ],
            "notes_from_basil": "الزاي للحركة والتقدم والزيادة. النقطة فوق الحرف توحي بنقطة على مسار الحركة."
        },
        "س": {
            "character": "س",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "لثوي",
                "articulation_method": "احتكاكي مهموس",
                "sound_echoes": ["صوت الهمس", "صوت الانسياب"],
                "general_sound_quality": "صوت انسيابي هامس"
            },
            "visual_form_semantics": ["خط متموج", "مسار متعرج"],
            "core_semantic_axes": {
                "flow_continuity": ("انسياب/استمرار/سلاسة", "توقف/تقطع")
            },
            "general_connotations": ["الانسياب", "الاستمرار", "السلاسة", "السير"],
            "examples_from_basil": [
                "سال، سيل: تفيد الانسياب والجريان",
                "سار، مسير: تفيد الحركة المستمرة",
                "سلس، سلاسة: تفيد السهولة والانسيابية"
            ],
            "notes_from_basil": "السين للانسياب والاستمرار والسلاسة. شكل الحرف المتموج يوحي بمسار انسيابي متعرج."
        },
        "ش": {
            "character": "ش",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "لثوي حنكي",
                "articulation_method": "احتكاكي مهموس",
                "sound_echoes": ["صوت التفشي", "صوت الانتشار"],
                "general_sound_quality": "صوت منتشر متفشي"
            },
            "visual_form_semantics": ["خط متموج مع نقاط", "انتشار وتفرع"],
            "core_semantic_axes": {
                "dispersion_branching": ("انتشار/تفرع/تشعب", "تجمع/تركيز")
            },
            "general_connotations": ["الانتشار", "التفرع", "التشعب", "التفشي"],
            "examples_from_basil": [
                "شجرة: تفيد التفرع والانتشار",
                "شع، إشعاع: تفيد الانتشار من مركز",
                "شرح، شرح: تفيد التوسع والتفصيل"
            ],
            "notes_from_basil": "الشين للانتشار والتفرع والتشعب. النقاط الثلاث فوق الحرف توحي بالانتشار والتفرع من أصل واحد."
        },
        "ص": {
            "character": "ص",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "أسناني لثوي مفخم",
                "articulation_method": "احتكاكي مهموس مفخم",
                "sound_echoes": ["صوت الصلابة", "صوت القوة"],
                "general_sound_quality": "صوت قوي مفخم"
            },
            "visual_form_semantics": ["دائرة مغلقة", "وعاء محكم"],
            "core_semantic_axes": {
                "solidity_purity": ("صلابة/نقاء/خلوص", "ليونة/تلوث")
            },
            "general_connotations": ["الصلابة", "النقاء", "الخلوص", "الإحكام"],
            "examples_from_basil": [
                "صلب، صلابة: تفيد القوة والمتانة",
                "صفا، صفاء: تفيد النقاء والخلوص",
                "صان، صيانة: تفيد الحفظ والحماية"
            ],
            "notes_from_basil": "الصاد للصلابة والنقاء والإحكام. شكل الحرف يوحي بدائرة مغلقة محكمة."
        },
        "ض": {
            "character": "ض",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "أسناني لثوي مفخم",
                "articulation_method": "انفجاري مجهور مفخم",
                "sound_echoes": ["صوت الضغط", "صوت القوة"],
                "general_sound_quality": "صوت قوي مفخم ضاغط"
            },
            "visual_form_semantics": ["دائرة مغلقة مع نقطة", "ضغط وقوة"],
            "core_semantic_axes": {
                "pressure_force": ("ضغط/قوة/إلزام", "ضعف/تراخي")
            },
            "general_connotations": ["الضغط", "القوة", "الإلزام", "الضرورة"],
            "examples_from_basil": [
                "ضغط، ضاغط: تفيد الضغط والقوة",
                "ضرب، ضارب: تفيد التأثير القوي",
                "ضرورة، اضطرار: تفيد الإلزام والحتمية"
            ],
            "notes_from_basil": "الضاد للضغط والقوة والإلزام. النقطة فوق الحرف توحي بنقطة الضغط والتأثير."
        },
        "ط": {
            "character": "ط",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "أسناني لثوي مفخم",
                "articulation_method": "انفجاري مهموس مفخم",
                "sound_echoes": ["صوت الطرق", "صوت الامتداد"],
                "general_sound_quality": "صوت قوي مفخم ممتد"
            },
            "visual_form_semantics": ["خط أفقي مع دائرة", "امتداد وإحاطة"],
            "core_semantic_axes": {
                "extension_encirclement": ("امتداد/إحاطة/طول", "قصر/محدودية")
            },
            "general_connotations": ["الامتداد", "الإحاطة", "الطول", "الشمول"],
            "examples_from_basil": [
                "طال، طويل: تفيد الامتداد والطول",
                "طاف، طواف: تفيد الدوران والإحاطة",
                "طبق، إطباق: تفيد الشمول والتغطية"
            ],
            "notes_from_basil": "الطاء للامتداد والإحاطة والشمول. شكل الحرف يوحي بامتداد أفقي مع إحاطة دائرية."
        },
        "ظ": {
            "character": "ظ",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "أسناني لثوي مفخم",
                "articulation_method": "احتكاكي مجهور مفخم",
                "sound_echoes": ["صوت الظهور", "صوت البروز"],
                "general_sound_quality": "صوت قوي مفخم بارز"
            },
            "visual_form_semantics": ["خط أفقي مع دائرة ونقطة", "ظهور وبروز"],
            "core_semantic_axes": {
                "appearance_prominence": ("ظهور/بروز/وضوح", "خفاء/غموض")
            },
            "general_connotations": ["الظهور", "البروز", "الوضوح", "الظل"],
            "examples_from_basil": [
                "ظهر، ظهور: تفيد البروز والوضوح",
                "ظل، ظلال: تفيد الانعكاس والتجلي",
                "ظن، ظنون: تفيد التصور والتخيل"
            ],
            "notes_from_basil": "الظاء للظهور والبروز والوضوح. النقطة فوق الحرف توحي بنقطة الظهور والبروز."
        },
        "ع": {
            "character": "ع",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "حلقي",
                "articulation_method": "احتكاكي مجهور",
                "sound_echoes": ["صوت العمق", "صوت الاتساع"],
                "general_sound_quality": "صوت عميق واسع"
            },
            "visual_form_semantics": ["عين مفتوحة", "فتحة واسعة"],
            "core_semantic_axes": {
                "depth_knowledge": ("عمق/معرفة/إدراك", "سطحية/جهل"),
                "width_comprehensiveness": ("اتساع/شمول/عموم", "ضيق/خصوص")
            },
            "general_connotations": ["العمق", "المعرفة", "الإدراك", "الاتساع", "الشمول", "العموم"],
            "examples_from_basil": [
                "علم، معرفة: تفيد الإدراك والفهم",
                "عمق، عميق: تفيد البعد والغور",
                "عم، عموم: تفيد الشمول والاتساع"
            ],
            "notes_from_basil": "العين للعمق والمعرفة والاتساع. شكل الحرف يشبه العين المفتوحة التي ترى وتدرك، والفتحة الواسعة التي تشمل وتحيط."
        },
        "غ": {
            "character": "غ",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "حلقي",
                "articulation_method": "احتكاكي مجهور",
                "sound_echoes": ["صوت الغرغرة", "صوت الغموض"],
                "general_sound_quality": "صوت عميق غامض"
            },
            "visual_form_semantics": ["عين مغلقة", "غطاء"],
            "core_semantic_axes": {
                "mystery_covering": ("غموض/ستر/تغطية", "وضوح/كشف")
            },
            "general_connotations": ["الغموض", "الستر", "التغطية", "الغياب"],
            "examples_from_basil": [
                "غطى، تغطية: تفيد الستر والإخفاء",
                "غاب، غياب: تفيد الاختفاء والبعد",
                "غمض، غموض: تفيد الإبهام وعدم الوضوح"
            ],
            "notes_from_basil": "الغين للغموض والستر والتغطية. النقطة فوق الحرف توحي بالعين المغلقة أو المغطاة."
        },
        "ف": {
            "character": "ف",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "شفوي أسناني",
                "articulation_method": "احتكاكي مهموس",
                "sound_echoes": ["صوت النفخ", "صوت الفصل"],
                "general_sound_quality": "صوت هوائي فاصل"
            },
            "visual_form_semantics": ["فم مفتوح", "فتحة"],
            "core_semantic_axes": {
                "separation_opening": ("فصل/فتح/فراغ", "وصل/إغلاق/امتلاء")
            },
            "general_connotations": ["الفصل", "الفتح", "الفراغ", "الانفصال"],
            "examples_from_basil": [
                "فتح، فاتح: تفيد الفتح والكشف",
                "فصل، فاصل: تفيد القطع والتمييز",
                "فرغ، فراغ: تفيد الخلو والسعة"
            ],
            "notes_from_basil": "الفاء للفصل والفتح والفراغ. شكل الحرف يشبه الفم المفتوح أو الفتحة."
        },
        "ق": {
            "character": "ق",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "لهوي",
                "articulation_method": "انفجاري مهموس",
                "sound_echoes": ["صوت القطع", "صوت القوة"],
                "general_sound_quality": "صوت قوي قاطع"
            },
            "visual_form_semantics": ["دائرة مع نقطتين", "قوة وثبات"],
            "core_semantic_axes": {
                "strength_decisiveness": ("قوة/حسم/قطع", "ضعف/تردد")
            },
            "general_connotations": ["القوة", "الحسم", "القطع", "الثبات"],
            "examples_from_basil": [
                "قطع، قاطع: تفيد الفصل الحاسم",
                "قوي، قوة: تفيد الشدة والمتانة",
                "قام، قيام: تفيد الثبات والاستقرار"
            ],
            "notes_from_basil": "القاف للقوة والحسم والقطع. النقطتان فوق الحرف توحيان بالثبات والقوة."
        },
        "ك": {
            "character": "ك",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "حنكي",
                "articulation_method": "انفجاري مهموس",
                "sound_echoes": ["صوت الكبت", "صوت الكتم"],
                "general_sound_quality": "صوت مكتوم محبوس"
            },
            "visual_form_semantics": ["كف مقبوضة", "إمساك"],
            "core_semantic_axes": {
                "restraint_possession": ("كبت/إمساك/احتواء", "إطلاق/ترك")
            },
            "general_connotations": ["الكبت", "الإمساك", "الاحتواء", "التشبيه"],
            "examples_from_basil": [
                "كبت، كاتم: تفيد الحبس والمنع",
                "كف، كفاف: تفيد الإمساك والاحتواء",
                "كأن، مثل: تفيد التشبيه والمماثلة"
            ],
            "notes_from_basil": "الكاف للكبت والإمساك والاحتواء والتشبيه. شكل الحرف يشبه الكف المقبوضة التي تمسك شيئاً."
        },
        "ل": {
            "character": "ل",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "لثوي",
                "articulation_method": "جانبي",
                "sound_echoes": ["صوت اللين", "صوت الانسياب"],
                "general_sound_quality": "صوت لين منساب"
            },
            "visual_form_semantics": ["خط منحني لأعلى", "امتداد وارتفاع"],
            "core_semantic_axes": {
                "attachment_belonging": ("التصاق/انتماء/ملكية", "انفصال/استقلال")
            },
            "general_connotations": ["الالتصاق", "الانتماء", "الملكية", "الاختصاص"],
            "examples_from_basil": [
                "لصق، التصاق: تفيد الارتباط والقرب",
                "له، لي: تفيد الملكية والاختصاص",
                "لأجل، لكي: تفيد التعليل والغاية"
            ],
            "notes_from_basil": "اللام للالتصاق والانتماء والملكية. شكل الحرف يوحي بالامتداد والارتفاع والاتصال."
        },
        "م": {
            "character": "م",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "شفوي",
                "articulation_method": "أنفي",
                "sound_echoes": ["صوت الضم", "صوت الإغلاق"],
                "general_sound_quality": "صوت مغلق ممتلئ"
            },
            "visual_form_semantics": ["دائرة مغلقة", "تجمع واكتمال"],
            "core_semantic_axes": {
                "completion_fullness": ("اكتمال/امتلاء/تمام", "نقص/فراغ")
            },
            "general_connotations": ["الاكتمال", "الامتلاء", "التمام", "الجمع"],
            "examples_from_basil": [
                "تم، تمام: تفيد الاكتمال والنهاية",
                "جمع، مجموع: تفيد الضم والتجميع",
                "ملأ، امتلاء: تفيد الشغل والتعبئة"
            ],
            "notes_from_basil": "الميم للاكتمال والامتلاء والتمام. شكل الحرف يوحي بالدائرة المغلقة المكتملة."
        },
        "ن": {
            "character": "ن",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "لثوي",
                "articulation_method": "أنفي",
                "sound_echoes": ["صوت الأنين", "صوت الرنين"],
                "general_sound_quality": "صوت رنان مستمر"
            },
            "visual_form_semantics": ["نقطة فوق حوض", "بذرة في تربة"],
            "core_semantic_axes": {
                "emergence_growth": ("نمو/ظهور/بروز", "كمون/خفاء")
            },
            "general_connotations": ["النمو", "الظهور", "البروز", "الاستمرار"],
            "examples_from_basil": [
                "نبت، نمو: تفيد الظهور والزيادة",
                "نور، إنارة: تفيد الإضاءة والوضوح",
                "نون: اسم الحرف يرتبط بالحوت والماء والحياة"
            ],
            "notes_from_basil": "النون للنمو والظهور والبروز. النقطة فوق الحرف توحي بالبذرة التي تنمو وتظهر."
        },
        "ه": {
            "character": "ه",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "حنجري",
                "articulation_method": "احتكاكي مهموس",
                "sound_echoes": ["صوت التنفس", "صوت الهواء"],
                "general_sound_quality": "صوت هوائي خفيف"
            },
            "visual_form_semantics": ["دائرة مفتوحة", "فراغ وهواء"],
            "core_semantic_axes": {
                "emptiness_lightness": ("فراغ/خفة/هواء", "امتلاء/ثقل")
            },
            "general_connotations": ["الفراغ", "الخفة", "الهواء", "الهدوء"],
            "examples_from_basil": [
                "هواء، تهوية: تفيد الخفة والفراغ",
                "هدأ، هدوء: تفيد السكون والراحة",
                "هاء: اسم الحرف يرتبط بالتنفس والحياة"
            ],
            "notes_from_basil": "الهاء للفراغ والخفة والهواء. شكل الحرف يوحي بالدائرة المفتوحة أو الفراغ."
        },
        "و": {
            "character": "و",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "شفوي",
                "articulation_method": "شبه حركة",
                "sound_echoes": ["صوت الوصل", "صوت الامتداد"],
                "general_sound_quality": "صوت ممتد موصول"
            },
            "visual_form_semantics": ["حلقة متصلة", "وصلة"],
            "core_semantic_axes": {
                "connection_continuity": ("وصل/ربط/استمرار", "فصل/قطع")
            },
            "general_connotations": ["الوصل", "الربط", "الاستمرار", "الجمع"],
            "examples_from_basil": [
                "وصل، واصل: تفيد الربط والاتصال",
                "ودام، دوام: تفيد الاستمرار والبقاء",
                "وجمع، مجموع: تفيد الضم والتجميع"
            ],
            "notes_from_basil": "الواو للوصل والربط والاستمرار. شكل الحرف يوحي بالحلقة المتصلة أو الوصلة."
        },
        "ي": {
            "character": "ي",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "حنكي",
                "articulation_method": "شبه حركة",
                "sound_echoes": ["صوت الامتداد", "صوت الليونة"],
                "general_sound_quality": "صوت لين ممتد"
            },
            "visual_form_semantics": ["خط منحني مع نقطتين", "يد ممدودة"],
            "core_semantic_axes": {
                "extension_possession": ("امتداد/ملكية/نسبة", "انقطاع/انفصال")
            },
            "general_connotations": ["الامتداد", "الملكية", "النسبة", "الإضافة"],
            "examples_from_basil": [
                "يد، أيدي: تفيد الامتداد والقدرة",
                "لي، إليّ: تفيد الملكية والنسبة",
                "يمن، يمين: تفيد القوة والبركة"
            ],
            "notes_from_basil": "الياء للامتداد والملكية والنسبة. شكل الحرف يوحي باليد الممدودة أو الخط المنحني الممتد."
        },
        "ء": {
            "character": "ء",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "حنجري",
                "articulation_method": "انفجاري حنجري",
                "sound_echoes": ["صوت المفاجأة", "صوت التوقف المفاجئ"],
                "general_sound_quality": "صوت مفاجئ قوي"
            },
            "visual_form_semantics": ["نقطة صغيرة", "توقف مفاجئ"],
            "core_semantic_axes": {
                "surprise_interruption": ("مفاجأة/توقف/قطع", "استمرار/تدفق")
            },
            "general_connotations": ["المفاجأة", "التوقف", "القطع", "البداية"],
            "examples_from_basil": [
                "سأل، مسألة: تفيد الاستفهام والمفاجأة",
                "بدأ، ابتداء: تفيد البدء والشروع",
                "قرأ، قراءة: تفيد النطق والتلفظ"
            ],
            "notes_from_basil": "الهمزة للمفاجأة والتوقف والقطع. شكل الحرف يوحي بالنقطة الصغيرة التي تمثل التوقف المفاجئ."
        },
        "ة": {
            "character": "ة",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "حنجري",
                "articulation_method": "احتكاكي مهموس",
                "sound_echoes": ["صوت الهمس الخفيف", "صوت النهاية"],
                "general_sound_quality": "صوت هامس خفيف"
            },
            "visual_form_semantics": ["وعاء صغير", "احتواء وتأنيث"],
            "core_semantic_axes": {
                "femininity_containment": ("تأنيث/احتواء/نهاية", "تذكير/انفتاح/استمرار")
            },
            "general_connotations": ["التأنيث", "الاحتواء", "النهاية", "التخصيص"],
            "examples_from_basil": [
                "معلمة، طالبة: تفيد التأنيث والتخصيص",
                "حديقة، غرفة: تفيد الاحتواء والإحاطة",
                "نهاية، خاتمة: تفيد الاكتمال والختام"
            ],
            "notes_from_basil": "التاء المربوطة للتأنيث والاحتواء والنهاية. شكل الحرف يوحي بالوعاء الصغير الذي يحتوي شيئاً ما."
        }
    },
    "en": {
        "A": {
            "character": "A",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت الطفل 'آه' لطلب الحنان والقرب",
                    "صيحة تعجب أو إنذار"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "قمة جبل",
                "برج",
                "رمز رأس الثور (Aleph)",
                "أذن تستمع",
                "ورقة شجر رقيقة",
                "حلقة (مثل القرط)",
                "خيمة أو مأوى"
            ],
            "core_semantic_axes": {
                "authority_tenderness": [
                    "جلال/طموح/سلطة/إعجاب/روعة (للشكل الكبير)",
                    "حنان/قرب/احتواء/دفء (للشكل الصغير)"
                ]
            },
            "general_connotations": [
                "Apex",
                "Ambition",
                "Authority",
                "Admiration",
                "Awesome",
                "affection",
                "adore",
                "accompany",
                "arm",
                "at home",
                "comfort, closeness, yearning for affection, security of embrace",
                "ascent, attainment of altitude, reaching higher"
            ],
            "examples_from_basil": [
                "When a baby cries with \"Aah\" sound, the underlying need is for comfort and closeness",
                "In the word \"Aim\", the 'A' evokes ascent and attainment of altitude"
            ],
            "notes_from_basil": "Does its triangular form not powerfully evoke the image of a mountain peak... suggesting strength and primacy? The open 'Aah' sound itself often forms the basis of exclamations... Does it not bring to mind the shape of an ear, listening intently? ...This feeling of containing, holding gently, seems present in words of closeness and care."
        },
        "O": {
            "character": "O",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت الدهشة والتعجب",
                    "صوت الدائرة المكتملة"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "دائرة كاملة",
                "عين مفتوحة",
                "فم متعجب",
                "دائرة صغيرة",
                "نقطة",
                "بذرة"
            ],
            "core_semantic_axes": {
                "wholeness_emptiness": [
                    "اكتمال/شمول/دائرية",
                    "فراغ/خواء/انفتاح"
                ],
                "movement_containment": [
                    "Forward motion and progression",
                    "encompassing or returning"
                ]
            },
            "general_connotations": [
                "Orb",
                "Omnipresent",
                "Overall",
                "Observe",
                "Open",
                "origin",
                "orbit",
                "only",
                "omen",
                "odd",
                "forward motion, progression, accompaniment"
            ],
            "examples_from_basil": [
                "The long \"Ooh\" sound signals a desire to move towards, follow, join in",
                "The letter appears in English words connected to impetus, advancing",
                "Sometimes associated with themes of encompassing or returning",
                "The perfect roundness evokes the wheel, symbol of movement"
            ],
            "notes_from_basil": "O is the perfect circle, the shape of completion and wholeness. It is the shape of the eye that observes, the mouth that exclaims in wonder, the cycle that returns to its beginning. The sound itself requires the mouth to form a perfect circle - a physical embodiment of its meaning."
        },
        "I": {
            "character": "I",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت الأنين الخافت",
                    "صوت الألم المكتوم"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "عمود مستقيم",
                "أداة حادة",
                "رمز للذات المنتصبة",
                "نقطة صغيرة مع خط",
                "علامة بسيطة",
                "شيء ينسحب إلى ذاته"
            ],
            "core_semantic_axes": {
                "inward_outward": [
                    "انطواء/كتمان/داخلية",
                    "تأكيد ذات/وضوح/ظهور"
                ]
            },
            "general_connotations": [
                "Identity",
                "Individual",
                "Introspection",
                "Insight",
                "Immediate",
                "inner",
                "intimate",
                "invisible",
                "inward",
                "insular",
                "inward feelings, psychological states"
            ],
            "examples_from_basil": [
                "The long \"Eee\" sound may suggest psychological discomfort or suppressed feeling",
                "In the word \"Aim\", the 'i' suggests the individual climber, standing in place"
            ],
            "notes_from_basil": "The lowercase 'i' can appear quite small, almost like a subtle mark, maybe even suggesting something retreating into itself, unseen. The uppercase 'I', by contrast, stands assertive and distinct, sometimes bringing to mind the straight, sharp form of a tool."
        },
        "R": {
            "character": "R",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت الدوران والتكرار",
                    "صوت الحركة المستمرة"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "رأس متحرك",
                "قوس مع ساق",
                "رمز للحركة المستمرة",
                "حركة دائرية غير مكتملة",
                "موجة صغيرة",
                "movement, partial rotation, or flowing energy"
            ],
            "core_semantic_axes": {
                "movement_resistance": [
                    "حركة/تدفق/استمرارية",
                    "مقاومة/توقف/انقطاع"
                ]
            },
            "general_connotations": [
                "Repeat",
                "Rotate",
                "Roll",
                "Rhythm",
                "Radiate",
                "recurring",
                "ripple",
                "roar",
                "run",
                "rapid",
                "forward motion and fluid repetition",
                "energy, vibration, and dynamic movement"
            ],
            "examples_from_basil": [
                "In the word \"Fire\", the 'r' represents roaring, radiating, flowing energy",
                "In the word "Bird", the 'r' signifies rapid, recurring flutter or trajectory",
                "The letter is described as "truly prominent and dynamic" in English"
            ],
            "notes_from_basil": "Let's turn our attention now to a truly prominent and dynamic letter in our English alphabet: the letter 'R'. As you rightly observed, its presence is remarkably widespread, making it a frequent and significant component in countless words across many languages, certainly including our own. However, applying the principle we've established – the letter representing a whole spectrum or scale – 'R' doesn't solely signify forward motion and fluid repetition. It also defines the axis upon which resistance to flow, cessation of movement, and disruption of rhythm exist. The very energy required for movement ('R') is also present in the force needed to stop or counter that movement."
        },
        "N": {
            "character": "N",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت الاتصال والارتباط",
                    "صوت الاستقرار"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "عمود مع قوس",
                "جسر",
                "اتصال بين نقطتين",
                "قوس واحد",
                "نصف حرف M"
            ],
            "core_semantic_axes": {
                "connection_separation": [
                    "اتصال/ارتباط/تواصل",
                    "انفصال/استقلالية"
                ]
            },
            "general_connotations": [
                "Network",
                "Node",
                "Nexus",
                "Navigate",
                "Near",
                "new",
                "next",
                "now",
                "name",
                "note",
                "connection, arrival, stability",
                "the concept of linking or bridging"
            ],
            "examples_from_basil": [
                "In the word "Name" - the identifier constantly on our lips and tongues",
                "In the word \"Aim\", the 'n' reinforces the sense of arrival, connection to the peak, or finding stability"
            ],
            "notes_from_basil": "Could this visual suggestion of 'n' being a part of 'm', or 'm' being a kind of doubling or expansion of 'n', hint at their semantic roles? You made the insightful point that both are articulated near the lips/front mouth, connecting them perhaps to outward expression. How fitting, then, that they combine in the word Name** – the very label we articulate, the identifier constantly on our lips and tongues!"
        },
        "M": {
            "character": "M",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت الشفاه المغلقة",
                    "صوت الهمهمة"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "قمم متعددة",
                "تموجات",
                "جبال",
                "قوسان متصلان",
                "موجتان",
                "تكرار"
            ],
            "core_semantic_axes": {
                "multiplicity_singularity": [
                    "تعدد/تكرار/كثرة",
                    "وحدة/فردية"
                ]
            },
            "general_connotations": [
                "Multiple",
                "Mountain",
                "Magnitude",
                "Mass",
                "Measure",
                "many",
                "more",
                "mother",
                "make",
                "move",
                "multiplicity, abundance, or magnitude",
                "maternal or nurturing concepts"
            ],
            "examples_from_basil": [
                "In the word "Name" - the very label we articulate, constantly on our lips and tongues",
                "The 'M' appears to be a doubling or expansion of 'N'"
            ],
            "notes_from_basil": "Could this visual suggestion of 'n' being a part of 'm', or 'm' being a kind of doubling or expansion of 'n', hint at their semantic roles? You made the insightful point that both are articulated near the lips/front mouth, connecting them perhaps to outward expression. How fitting, then, that they combine in the word Name** – the very label we articulate, the identifier constantly on our lips and tongues!"
        },
        "W": {
            "character": "W",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت التموج والتذبذب",
                    "صوت الحركة المتكررة"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "تموجات متعددة",
                "خط متعرج",
                "جبال وواديان",
                "تموج مزدوج",
                "حركة متكررة"
            ],
            "core_semantic_axes": {
                "undulation_stability": [
                    "تموج/تذبذب/تكرار",
                    "ثبات/استقرار"
                ]
            },
            "general_connotations": [
                "Wave",
                "Weave",
                "Wind",
                "Wander",
                "Work",
                "water",
                "wind",
                "way",
                "walk",
                "weave",
                "undulating movement, waves, or repetitive motion",
                "multiplicity or collection (crowd, weight)",
                "processes involving back-and-forth movement"
            ],
            "examples_from_basil": [
                "In the word "Water" - starts with the undulating 'w', instantly suggesting waves or fluidity",
                "Associated with concepts like "crowd" (multiplicity contained)",
                "Connected to activities like "sewing/weaving" (continuous movement)"
            ],
            "notes_from_basil": "Your idea that gathering people requires a boundary, perhaps imagined as a jagged or multi-pointed line like 'W', is intriguing. A crowd is a multiplicity contained, a collection of points. It might also suggest the weight or undulating movement of a large group, like a swarm. Beyond these initial examples, the 'W' shape strongly evokes repetitive, back-and-forth, or undulating motion: the process of stitching or weaving involves a continuous movement, guiding the thread through the fabric in a pattern that can resemble the 'W's path."
        },
        "S": {
            "character": "S",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت الانسياب والتدفق",
                    "صوت الهسهسة"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "ثعبان",
                "نهر متعرج",
                "مسار متموج",
                "انحناءة",
                "موجة صغيرة",
                "حركة سلسة"
            ],
            "core_semantic_axes": {
                "flow_resistance": [
                    "انسياب/تدفق/سلاسة",
                    "مقاومة/احتكاك"
                ]
            },
            "general_connotations": [
                "Stream",
                "Smooth",
                "Serpentine",
                "Slide",
                "Sway",
                "soft",
                "silk",
                "swift",
                "soothe",
                "subtle",
                "smoothness, flow, or continuous movement",
                "sounds that sustain or continue"
            ],
            "examples_from_basil": [
                "In the word "Sing" - the letter 'S', with its flowing sound and undulating shape, suggests smoothness or movement",
                "In the word "Sail" - the 's' suggests the curve of the wind or water"
            ],
            "notes_from_basil": "You offered a beautiful insight here. The letter 'S', with its flowing sound and undulating shape, already suggests smoothness or movement. Combine this with the sustained resonance of '-ing' /ŋ/, and we perfectly capture the essence of singing – a continuous, flowing, resonant expression. The visual 'S' adds that lovely hint of rhythm or even dance."
        },
        "T": {
            "character": "T",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت الضربة أو النقرة",
                    "صوت الوقف المفاجئ"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "مطرقة",
                "فأس",
                "أداة بناء أو هدم",
                "صليب",
                "مطرقة صغيرة",
                "عصا",
                "أداة",
                "a solid, stable structure or foundation"
            ],
            "core_semantic_axes": {
                "construction_destruction": [
                    "بناء/تأسيس/تثبيت",
                    "هدم/تفكيك/كسر"
                ]
            },
            "general_connotations": [
                "Tool",
                "Top",
                "Tower",
                "Tall",
                "Touch",
                "tap",
                "take",
                "turn",
                "time",
                "tell",
                "tools, building, or construction",
                "stability, foundation, or structure"
            ],
            "examples_from_basil": [
                "In the word "Water" - the 't' might hint at a point of stability or containment",
                "The uppercase 'T' strikingly resembles a hammer or pickaxe"
            ],
            "notes_from_basil": "This connection isn't limited to sound alone. Visually, as you keenly observed, the uppercase 'T' strikingly resembles a hammer or perhaps a pickaxe – fundamental tools of both building and breaking down."
        },
        "B": {
            "character": "B",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت الانفجار الشفوي",
                    "صوت البداية"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "كتلة",
                "حاجز",
                "عنصر أساسي",
                "وعاء",
                "كرة مع عمود",
                "بذرة نابتة",
                "جنين"
            ],
            "core_semantic_axes": {
                "beginning_ending": [
                    "بداية/انطلاق/ولادة",
                    "حد/حاجز/إغلاق"
                ]
            },
            "general_connotations": [
                "Begin",
                "Build",
                "Barrier",
                "Base",
                "Birth",
                "baby",
                "bud",
                "burst",
                "bubble",
                "ball",
                "beginnings, birth, or bursting forth",
                "foundational elements or building blocks"
            ],
            "examples_from_basil": [
                "In the word "Bird" - begins with 'b' (perhaps the initial burst of flight, the rounded body, or the beak)",
                "The solid, curved forms might evoke images of blocks, barriers, or containers"
            ],
            "notes_from_basil": "While perhaps less direct, the solid, often curved or enclosed forms of 'B' and 'D' might also evoke images of blocks, barriers, foundational elements, or perhaps even the shape of a shovel's scoop ('d') or a container ('B')."
        },
        "D": {
            "character": "D",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت الوقف الشفوي",
                    "صوت الحد أو النهاية"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "باب",
                "قوس مغلق",
                "حاجز نصف دائري",
                "مجرفة",
                "وعاء صغير",
                "نصف دائرة مع عمود"
            ],
            "core_semantic_axes": {
                "definition_ambiguity": [
                    "تحديد/تعريف/وضوح",
                    "غموض/تداخل"
                ]
            },
            "general_connotations": [
                "Define",
                "Door",
                "Divide",
                "Depth",
                "Destination",
                "deep",
                "down",
                "dig",
                "done",
                "direct",
                "definition, demarcation, or boundaries",
                "depth, digging, or downward movement",
                "doorways, destinations, or endpoints"
            ],
            "examples_from_basil": [
                "In the word "Bird" - ends with 'd', forming a visual pairing with 'b' that captures dynamic motion",
                "The solid, curved form might evoke images of blocks, barriers, or containers"
            ],
            "notes_from_basil": "While perhaps less direct, the solid, often curved or enclosed forms of 'B' and 'D' might also evoke images of blocks, barriers, foundational elements, or perhaps even the shape of a shovel's scoop ('d') or a container ('B'). And let's recall our principle that letter shapes often function as multifaceted symbols ('رمز بين'), capable of evoking multiple related images or actions within a conceptual field."
        },
        "F": {
            "character": "F",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت النفخ أو الهواء المندفع",
                    "صوت الاحتكاك"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "علم",
                "عمود مع أذرع",
                "شوكة",
                "خطاف",
                "منجل صغير",
                "أداة معلقة",
                "a tool with extensions or projections"
            ],
            "core_semantic_axes": {
                "flow_obstruction": [
                    "تدفق/انطلاق/حرية",
                    "احتكاك/مقاومة"
                ]
            },
            "general_connotations": [
                "Flow",
                "Force",
                "Fan",
                "Forward",
                "Free",
                "fire",
                "flight",
                "flutter",
                "float",
                "flicker",
                "flow, flight, or forward movement",
                "fire, flame, or combustion",
                "fanning, fluttering, or fluctuation"
            ],
            "examples_from_basil": [
                "In the word \"Fire\" - begins with the 'f' sound, reminiscent of a puff of air, fanning the flames or the sound of combustion itself"
            ],
            "notes_from_basil": "Consider fire. Does it not begin with the 'f' sound, reminiscent of a puff of air, fanning the flames or the sound of combustion itself?"
        },
        "E": {
            "character": "E",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت الانفتاح المتوسط",
                    "صوت الراحة والاسترخاء"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "مشط",
                "سلم أفقي",
                "أداة متعددة الأسنان",
                "نصف دائرة مفتوحة",
                "فم مبتسم",
                "كأس",
                "extension, reaching out, or embracing"
            ],
            "core_semantic_axes": {
                "extension_contraction": [
                    "امتداد/انتشار/توسع",
                    "تقلص/تركيز"
                ]
            },
            "general_connotations": [
                "Extend",
                "Equal",
                "Embrace",
                "Expand",
                "Establish",
                "ease",
                "even",
                "echo",
                "edge",
                "emerge",
                "extension, expansion, or reaching outward",
                "equilibrium, equality, or balance",
                "emergence, establishment, or foundation"
            ],
            "examples_from_basil": [
                "The letter appears in words related to extension, equilibrium, and establishment",
                "Suggests a middle ground or balanced state"
            ],
            "notes_from_basil": "The letter E represents extension and equilibrium - a middle ground that is neither fully open like A nor closed like I. Its shape suggests reaching out, embracing, or establishing foundations with multiple levels."
        },
        "Y": {
            "character": "Y",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت التساؤل والتعجب",
                    "صوت الانفتاح المائل"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "طريق متفرع",
                "غصن شجرة",
                "مفترق طرق",
                "كأس",
                "وعاء",
                "جذر ممتد للأسفل",
                "divergence, branching, or choice",
                "a path that splits or divides"
            ],
            "core_semantic_axes": {
                "division_unity": [
                    "تفرع/انقسام/خيار",
                    "تجمع/التقاء"
                ]
            },
            "general_connotations": [
                "Yield",
                "Yearn",
                "Yoke",
                "Year",
                "Youth",
                "yes",
                "young",
                "yell",
                "yarn",
                "yawn",
                "youth, yearning, or potential",
                "yielding, bending, or flexibility",
                "divergence, choice, or multiple paths"
            ],
            "examples_from_basil": [
                "The letter appears in words related to questioning, youth, and potential",
                "Suggests a branching path or decision point"
            ],
            "notes_from_basil": "The letter Y represents a point of divergence or choice - like a path that splits into two directions. Its shape mirrors this meaning perfectly, showing a single stem that branches into two paths."
        },
        "Z": {
            "character": "Z",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت الاحتكاك المستمر",
                    "صوت القطع المتتابع"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "برق",
                "مسار متعرج",
                "سكين مسننة",
                "خط متعرج",
                "موجة حادة",
                "حركة سريعة",
                "lightning, cutting tools, or rapid movement"
            ],
            "core_semantic_axes": {
                "energy_rest": [
                    "نشاط/حيوية/حركة",
                    "سكون/نهاية/اكتمال"
                ]
            },
            "general_connotations": [
                "Zeal",
                "Zigzag",
                "Zone",
                "Zenith",
                "Zero",
                "zip",
                "zap",
                "zoom",
                "zest",
                "zone",
                "energy, zeal, or zealousness",
                "zigzag movements, lightning, or electricity",
                "completion, finality (last letter of alphabet)"
            ],
            "examples_from_basil": [
                "The letter appears in words related to energy, rapid movement, and finality",
                "Suggests sharp, decisive action or completion"
            ],
            "notes_from_basil": "The letter Z, with its sharp angles and zigzag shape, evokes lightning, energy, and decisive action. As the final letter of our alphabet, it also carries connotations of completion or finality."
        },
        "L": {
            "character": "L",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت الانسياب اللساني",
                    "صوت الليونة والسلاسة"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "زاوية",
                "عمود مع قاعدة",
                "أداة قياس",
                "خط مستقيم مع انحناء",
                "عصا",
                "أداة",
                "stability, foundation, or support",
                "a right angle or measuring tool"
            ],
            "core_semantic_axes": {
                "flow_structure": [
                    "انسياب/ليونة/سلاسة",
                    "ثبات/دعم/هيكل"
                ]
            },
            "general_connotations": [
                "Long",
                "Level",
                "Lift",
                "Line",
                "Light",
                "love",
                "life",
                "low",
                "lay",
                "link",
                "length, extension, or linearity",
                "levels, layers, or foundations",
                "love, life, or fundamental concepts"
            ],
            "examples_from_basil": [
                "The letter appears in words related to fundamental concepts (love, life)",
                "Suggests both flowing movement and structural support"
            ],
            "notes_from_basil": "The letter L combines the stability of its vertical line with the flow suggested by its sound - representing both structure and movement, foundation and flexibility."
        },
        "C": {
            "character": "C",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت الاحتكاك أو القطع",
                    "صوت الانغلاق الجزئي"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "هلال",
                "وعاء مفتوح",
                "كأس",
                "قوس صغير",
                "منحنى",
                "خطاف",
                "a partial enclosure or container"
            ],
            "core_semantic_axes": {
                "containment_release": [
                    "احتواء/تجميع/ضم",
                    "انفتاح/إطلاق"
                ]
            },
            "general_connotations": [
                "Contain",
                "Cut",
                "Curve",
                "Cup",
                "Collect",
                "catch",
                "carry",
                "create",
                "circle",
                "cover",
                "containment, collection, or cupping",
                "cutting, carving, or creating",
                "curves, circles, or cyclical patterns"
            ],
            "examples_from_basil": [
                "The curved shape suggests containment or collection",
                "Dual nature (hard/soft sound) reflects versatility"
            ],
            "notes_from_basil": "The letter C, with its curved form, suggests containment or collection - like a cup or vessel that holds something. Its dual sound nature (hard as in 'cut' or soft as in 'circle') reflects its versatility in representing both separation and connection."
        },
        "P": {
            "character": "P",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت الانفجار الشفوي",
                    "صوت الضغط والإطلاق"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "علم مع عمود",
                "مطرقة",
                "أداة ذات رأس",
                "خطاف معكوس",
                "مفتاح",
                "أداة تعليق",
                "a tool with a head or projection",
                "something standing upright with support"
            ],
            "core_semantic_axes": {
                "pressure_release": [
                    "ضغط/دفع/قوة",
                    "إطلاق/تحرير"
                ]
            },
            "general_connotations": [
                "Push",
                "Point",
                "Project",
                "Place",
                "Power",
                "path",
                "put",
                "pull",
                "press",
                "precise",
                "pressure, pushing, or projection",
                "placement, positioning, or precision",
                "paths, processes, or procedures"
            ],
            "examples_from_basil": [
                "The letter appears in words related to pressure, precision, and placement",
                "Suggests both forceful action and careful positioning"
            ],
            "notes_from_basil": "The letter P combines the explosive energy of its sound with the precision of its shape - representing both forceful projection and careful placement."
        },
        "H": {
            "character": "H",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت النفس أو الهواء",
                    "صوت التنفس"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "سلم",
                "جسر",
                "بوابة",
                "كرسي",
                "مقعد",
                "منصة صغيرة",
                "connection, bridging, or support",
                "a structure with two vertical supports"
            ],
            "core_semantic_axes": {
                "height_depth": [
                    "ارتفاع/علو/سمو",
                    "عمق/أساس/جذر"
                ]
            },
            "general_connotations": [
                "Height",
                "Home",
                "Hold",
                "Human",
                "Heart",
                "help",
                "hand",
                "here",
                "high",
                "hope",
                "height, home, or humanity",
                "help, holding, or supporting",
                "heart, hope, or fundamental human concepts"
            ],
            "examples_from_basil": [
                "The letter appears in words related to fundamental human concepts",
                "Suggests both elevation and foundation"
            ],
            "notes_from_basil": "The letter H, with its bridge-like structure, connects and supports - like a home that shelters or hands that help. Its breath-like sound reminds us of the essential human act of breathing."
        },
        "G": {
            "character": "G",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت الاحتكاك الحلقي",
                    "صوت الانغلاق الجزئي"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "دائرة مع ذراع",
                "خطاف كبير",
                "أداة جمع",
                "حلقة مع ذيل",
                "خطاف صغير",
                "أداة تعليق",
                "a circular motion with direction or purpose"
            ],
            "core_semantic_axes": {
                "growth_decline": [
                    "نمو/تطور/ازدهار",
                    "تراجع/انحدار"
                ]
            },
            "general_connotations": [
                "Grow",
                "Ground",
                "Gather",
                "Give",
                "Great",
                "gain",
                "good",
                "guide",
                "group",
                "goal",
                "growth, grounding, or gravity",
                "gathering, grouping, or guiding",
                "generosity, giving, or goodness"
            ],
            "examples_from_basil": [
                "The curved shape suggests growth or gathering",
                "Dual nature (hard/soft sound) reflects versatility in meaning"
            ],
            "notes_from_basil": "The letter G combines the grounding stability of its shape with the growth potential of its sound - representing both foundation and development."
        },
        "K": {
            "character": "K",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت الانفجار الحلقي",
                    "صوت الوقف الحاد"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "زاوية حادة مع ساق",
                "سكين",
                "أداة قطع",
                "خط منكسر",
                "زاوية صغيرة",
                "أداة حادة",
                "a tool with cutting or pointing function"
            ],
            "core_semantic_axes": {
                "sharpness_smoothness": [
                    "حدة/دقة/قطع",
                    "نعومة/انسياب"
                ]
            },
            "general_connotations": [
                "Key",
                "Know",
                "Keep",
                "Keen",
                "Knot",
                "knife",
                "keen",
                "kill",
                "kind",
                "knack",
                "knowledge, keys, or unlocking",
                "sharpness, cutting, or precision",
                "keeping, containing, or preserving"
            ],
            "examples_from_basil": [
                "The angular shape suggests sharpness or precision",
                "Represents both cutting action and key-like function"
            ],
            "notes_from_basil": "The letter K combines the decisive sharpness of its sound with the angular precision of its shape - representing both cutting action and key-like function that unlocks knowledge."
        },
        "J": {
            "character": "J",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت الاحتكاك المركب",
                    "صوت القفز أو الانطلاق"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "خطاف كبير",
                "عصا مع انحناء",
                "أداة تعليق",
                "خط مع ذيل للأسفل",
                "خطاف معكوس",
                "أداة غوص",
                "a tool for joining or a path with a sudden drop"
            ],
            "core_semantic_axes": {
                "joining_separating": [
                    "ربط/وصل/التقاء",
                    "قفز/انفصال"
                ]
            },
            "general_connotations": [
                "Join",
                "Jump",
                "Journey",
                "Joy",
                "Jolt",
                "join",
                "jut",
                "jab",
                "jet",
                "jerk",
                "joining, junction, or juxtaposition",
                "jumping, jolting, or sudden movement",
                "journeys, joy, or jubilation"
            ],
            "examples_from_basil": [
                "The hook-like shape suggests joining or catching",
                "Represents both connection and sudden movement"
            ],
            "notes_from_basil": "The letter J combines the joining function of its hook-like shape with the jumping energy of its sound - representing both connection and dynamic movement."
        },
        "Q": {
            "character": "Q",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت الانغلاق الحلقي المتبوع بصوت آخر",
                    "صوت الاستفهام"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "دائرة مع ذيل",
                "مفتاح",
                "أداة فتح",
                "دائرة صغيرة مع خط للأسفل",
                "مفتاح صغير"
            ],
            "core_semantic_axes": {
                "question_answer": [
                    "استفهام/بحث/تساؤل",
                    "إجابة/يقين"
                ]
            },
            "general_connotations": [
                "Question",
                "Quest",
                "Quality",
                "Queen",
                "Quiet",
                "query",
                "quench",
                "quick",
                "quaint",
                "quiver",
                "questioning, querying, or quests",
                "quality, quintessence, or core essence",
                "quietude, quiescence, or calm"
            ],
            "examples_from_basil": [
                "The circle with tail shape suggests both completion and direction",
                "Almost always paired with 'u', suggesting partnership or completion"
            ],
            "notes_from_basil": "The letter Q, always seeking its partner U, represents the quest for completion - the circle of its shape suggesting wholeness, while its tail points toward a direction or purpose."
        },
        "U": {
            "character": "U",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت الضم أو الاحتواء",
                    "صوت العمق والامتداد"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "وعاء",
                "كأس",
                "حوض",
                "حرف مفتوح للأعلى",
                "وعاء صغير",
                "a space that can hold or contain"
            ],
            "core_semantic_axes": {
                "unity_division": [
                    "وحدة/اتحاد/تجميع",
                    "انفصال/تفرق"
                ]
            },
            "general_connotations": [
                "Unity",
                "Under",
                "Ultimate",
                "Universal",
                "Uphold",
                "unite",
                "use",
                "understand",
                "upward",
                "utter",
                "unity, union, or unification",
                "understanding, underlying, or foundation",
                "upward movement or elevation"
            ],
            "examples_from_basil": [
                "Often paired with 'Q', suggesting partnership or completion",
                "The U-shape naturally suggests a container or vessel"
            ],
            "notes_from_basil": "The letter U, with its cup-like shape, naturally suggests a vessel that can contain or hold - representing both unity and depth."
        },
        "V": {
            "character": "V",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت الاحتكاك الشفوي",
                    "صوت الانقسام أو التفرع"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "سهم للأسفل",
                "شكل متفرع",
                "مثلث مقلوب",
                "حرف صغير متفرع للأسفل",
                "شوكة صغيرة",
                "a path that narrows or focuses downward"
            ],
            "core_semantic_axes": {
                "division_convergence": [
                    "انقسام/تفرع/تشعب",
                    "تركيز/تجمع"
                ]
            },
            "general_connotations": [
                "Victory",
                "Vigor",
                "Voice",
                "Value",
                "Vision",
                "vibrate",
                "vital",
                "vivid",
                "venture",
                "verify",
                "victory, vigor, or vitality",
                "voice, vocalization, or vibration",
                "vision, view, or perspective"
            ],
            "examples_from_basil": [
                "The V-shape naturally suggests division or direction",
                "Represents both downward focus and upward victory gesture"
            ],
            "notes_from_basil": "The letter V, with its distinctive shape pointing downward, suggests focus, direction, and division - yet when inverted as in the victory gesture, it represents triumph and elevation."
        },
        "X": {
            "character": "X",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "",
                "articulation_method": "",
                "sound_echoes": [
                    "صوت التقاطع أو الاحتكاك المزدوج",
                    "صوت الغموض"
                ],
                "general_sound_quality": ""
            },
            "visual_form_semantics": [
                "تقاطع",
                "علامة إلغاء",
                "إشارة خطأ",
                "تقاطع صغير",
                "علامة ضرب",
                "meeting point, target, or unknown variable"
            ],
            "core_semantic_axes": {
                "known_unknown": [
                    "غموض/مجهول/سر",
                    "تحديد/تعيين/وضوح"
                ]
            },
            "general_connotations": [
                "X-ray",
                "Xenon",
                "Xylophone",
                "Xerox",
                "Xeno",
                "cross",
                "exact",
                "extra",
                "exit",
                "extreme",
                "unknown, mystery, or exploration",
                "crossing, intersection, or meeting point",
                "exactness, precision, or target"
            ],
            "examples_from_basil": [
                "The X-shape naturally suggests intersection or crossing",
                "Often used to mark a spot or represent an unknown variable"
            ],
            "notes_from_basil": "The letter X, with its intersecting lines, represents both meeting point and mystery - marking the spot where paths cross while also symbolizing the unknown or unexplored."
        }
    }
}
