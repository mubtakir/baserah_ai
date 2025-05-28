# -*- coding: utf-8 -*-

initial_letter_semantics_data = {
    "ar": {
        "ا": {
            "character": "ا",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "حنجري",
                "articulation_method": "مد",
                "sound_echoes": ['صوت المفاجأة (مع الهمزة)', 'التعجب (مع الهمزة)'],
                "general_sound_quality": "صوت أساسي، مفتوح",
            },
            "visual_form_semantics": ['الاستقامة', 'الارتفاع', 'العلو', 'البداية'],
            "core_semantic_axes": {
                "magnitude_elevation": ('عظمة/ارتفاع/علو (حسي ومعنوي)', 'صغر/انخفاض'),
            },
            "general_connotations": ['العظمة', 'الارتفاع', 'العلو (الحسي والمعنوي)'],
            "examples_from_basil": [
                "أ ل م (في سورة البقرة): الألف للعظمة والتعظيم.",
            ],
            "notes_from_basil": "الألف يوحي للعظمة والارتفاع والعلو. الهمزة: صوت المفاجأة والرعب والصدمة وللتعجب. حروف العلة (ا, و, ي) مع الهمزة تفيد التعجب والفزع والخوف والفرح."
        },
        "ب": {
            "character": "ب",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "شفوي",
                "articulation_method": "انفجاري",
                "sound_echoes": ['صوت ارتطام', 'صوت امتلاء'],
                "general_sound_quality": "صوت ارتطام وامتلاء وتشبع",
            },
            "visual_form_semantics": ['حوض أو إناء (الشكل القديم كمربع)', 'بوابة (لأنه شفوي، مقدمة الفم)'],
            "core_semantic_axes": {
                "containment_transfer": ('امتلاء/تشبع/حمل/نقل', 'إفراغ/ترك'),
            },
            "general_connotations": ['الامتلاء', 'التشبع', 'النقل (سبب منطقي للامتلاء)'],
            "examples_from_basil": [
                "بحر/نهر: نقطة الباء السفلية توحي بقطرة ماء انصبت.",
                "طلب، حلب، سلب، نهب: فيها معنى الانتقال.",
                "بلع، بلغ، بعد، قرب: تفيد الانتقال والتشبع.",
                "اسم الحرف 'باء' من باء يبوء (امتلأ به وبان عليه).",
            ],
            "notes_from_basil": "الباء للامتلاء والتشبع والنقل. المعاني ترتبط ارتباط سببي ومنطقي. الحروف الشفوية كأنها حروف مادية ترسم الواقع العملياتي والحركي الملموس."
        },
        "ت": {
            "character": "ت",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "أسناني لثوي",
                "articulation_method": "انفجاري مهموس",
                "sound_echoes": ['صوت النقر الخفيف', 'صوت الطرق الناعم'],
                "general_sound_quality": "صوت خفيف ناعم",
            },
            "visual_form_semantics": ['نقطتان فوق خط أفقي', 'علامة صغيرة'],
            "core_semantic_axes": {
                "completion_continuation": ('إتمام/إكمال/تحقق', 'بداية/استمرار'),
            },
            "general_connotations": ['الإتمام', 'التحقق', 'الاكتمال', 'التأنيث'],
            "examples_from_basil": [
                "تم، تمام، أتم: تفيد الإتمام والاكتمال",
                "تاب، توب: تفيد الرجوع والعودة إلى الأصل",
            ],
            "notes_from_basil": "التاء للإتمام والتحقق والاكتمال. وهي أيضاً علامة التأنيث في اللغة العربية. التاء المربوطة (ة) تفيد الاحتواء والإحاطة والتأنيث."
        },
        "ث": {
            "character": "ث",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "أسناني",
                "articulation_method": "احتكاكي مهموس",
                "sound_echoes": ['صوت النفث', 'صوت الانتشار'],
                "general_sound_quality": "صوت انتشار وتفرق",
            },
            "visual_form_semantics": ['ثلاث نقاط فوق خط أفقي', 'انتشار وتعدد'],
            "core_semantic_axes": {
                "dispersion_collection": ('انتشار/تفرق/تعدد', 'تجمع/تركيز'),
            },
            "general_connotations": ['الانتشار', 'التفرق', 'التعدد', 'الكثرة'],
            "examples_from_basil": [
                "ثر، ثرى، ثروة: تفيد الكثرة والانتشار",
                "بث، نفث: تفيد النشر والتفريق",
            ],
            "notes_from_basil": "الثاء للانتشار والتفرق والتعدد. النقاط الثلاث فوق الحرف توحي بالتعدد والانتشار."
        },
        "ج": {
            "character": "ج",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "وسط الحنك",
                "articulation_method": "مركب",
                "sound_echoes": ['صوت التجمع', 'صوت الاحتواء'],
                "general_sound_quality": "صوت تجمع واحتواء",
            },
            "visual_form_semantics": ['وعاء أو حوض', 'تجويف'],
            "core_semantic_axes": {
                "containment_gathering": ('احتواء/تجمع/جمع', 'تفرق/انتشار'),
            },
            "general_connotations": ['الاحتواء', 'التجمع', 'الجمع', 'التجويف'],
            "examples_from_basil": [
                "جمع، جماعة: تفيد التجمع والاجتماع",
                "جوف، جيب: تفيد التجويف والاحتواء",
                "جبل: تجمع الصخور والتراب في مكان مرتفع",
            ],
            "notes_from_basil": "الجيم للاحتواء والتجمع والتجويف. شكل الحرف يشبه الوعاء أو الحوض الذي يحتوي شيئاً ما."
        },
        "ح": {
            "character": "ح",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "حلقي",
                "articulation_method": "احتكاكي مهموس",
                "sound_echoes": ['صوت التنفس العميق', 'صوت الحياة'],
                "general_sound_quality": "صوت عميق من الحلق",
            },
            "visual_form_semantics": ['دائرة مفتوحة', 'حدود', 'إحاطة'],
            "core_semantic_axes": {
                "life_boundary": ('حياة/حيوية/حركة', 'موت/سكون'),
                "containment_limitation": ('إحاطة/حدود/حماية', 'انفتاح/تجاوز'),
            },
            "general_connotations": ['الحياة', 'الحيوية', 'الحركة', 'الإحاطة', 'الحدود', 'الحماية'],
            "examples_from_basil": [
                "حي، حياة: تفيد الحياة والحيوية",
                "حوط، حاط: تفيد الإحاطة والحماية",
                "حد، حدود: تفيد التحديد والفصل",
            ],
            "notes_from_basil": "الحاء للحياة والحيوية والإحاطة. صوت الحاء يشبه صوت التنفس العميق الذي هو أساس الحياة. شكل الحرف يوحي بالإحاطة والحدود."
        },
        "خ": {
            "character": "خ",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "حلقي",
                "articulation_method": "احتكاكي مجهور",
                "sound_echoes": ['صوت الخروج', 'صوت النفاذ'],
                "general_sound_quality": "صوت خشن من الحلق",
            },
            "visual_form_semantics": ['دائرة مفتوحة مع نقطة', 'ثقب أو فتحة'],
            "core_semantic_axes": {
                "penetration_exit": ('خروج/نفاذ/اختراق', 'دخول/بقاء'),
            },
            "general_connotations": ['الخروج', 'النفاذ', 'الاختراق', 'الفراغ'],
            "examples_from_basil": [
                "خرج، خروج: تفيد الخروج والانفصال",
                "خرق، اختراق: تفيد النفاذ والاختراق",
                "خلا، خلاء: تفيد الفراغ والخلو",
            ],
            "notes_from_basil": "الخاء للخروج والنفاذ والاختراق. النقطة فوق الحرف توحي بثقب أو فتحة للخروج."
        },
        "د": {
            "character": "د",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "أسناني لثوي",
                "articulation_method": "انفجاري مجهور",
                "sound_echoes": ['صوت الدق', 'صوت الضرب'],
                "general_sound_quality": "صوت قوي حاد",
            },
            "visual_form_semantics": ['قوس مغلق', 'باب'],
            "core_semantic_axes": {
                "entry_access": ('دخول/ولوج/وصول', 'خروج/انفصال'),
            },
            "general_connotations": ['الدخول', 'الولوج', 'الوصول', 'الباب'],
            "examples_from_basil": [
                "دخل، دخول: تفيد الدخول والولوج",
                "درب، درج: تفيد المسار والطريق",
                "دار: تفيد المكان المحيط والمغلق",
            ],
            "notes_from_basil": "الدال للدخول والولوج والوصول. شكل الحرف يشبه الباب أو المدخل."
        },
        "ذ": {
            "character": "ذ",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "أسناني",
                "articulation_method": "احتكاكي مجهور",
                "sound_echoes": ['صوت الذوبان', 'صوت الانتشار اللطيف'],
                "general_sound_quality": "صوت انسيابي لين",
            },
            "visual_form_semantics": ['خط مع نقطة', 'إشارة'],
            "core_semantic_axes": {
                "indication_reference": ('إشارة/تذكير/ذكر', 'نسيان/إهمال'),
            },
            "general_connotations": ['الإشارة', 'التذكير', 'الذكر', 'الانتشار اللطيف'],
            "examples_from_basil": [
                "ذكر، تذكير: تفيد الذكر والتذكير",
                "ذاب، ذوبان: تفيد الانتشار والتلاشي اللطيف",
                "ذهب: تفيد الانتقال والمضي",
            ],
            "notes_from_basil": "الذال للإشارة والتذكير والانتشار اللطيف. النقطة فوق الحرف توحي بالإشارة والتنبيه."
        },
        "ر": {
            "character": "ر",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "لثوي",
                "articulation_method": "تكراري",
                "sound_echoes": ['صوت التكرار', 'صوت الحركة المستمرة'],
                "general_sound_quality": "صوت متكرر متحرك",
            },
            "visual_form_semantics": ['رأس منحني', 'حركة دائرية'],
            "core_semantic_axes": {
                "repetition_movement": ('تكرار/حركة/استمرارية', 'توقف/ثبات'),
            },
            "general_connotations": ['التكرار', 'الحركة', 'الاستمرارية', 'الدوران'],
            "examples_from_basil": [
                "كرر، تكرار: تفيد التكرار والإعادة",
                "دار، دوران: تفيد الحركة الدائرية",
                "جرى، جريان: تفيد الحركة المستمرة",
            ],
            "notes_from_basil": "الراء للتكرار والحركة والاستمرارية. صوت الراء متكرر بطبيعته، وشكل الحرف يوحي بالحركة الدائرية."
        },
        "ز": {
            "character": "ز",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "لثوي",
                "articulation_method": "احتكاكي مجهور",
                "sound_echoes": ['صوت الزحف', 'صوت الانزلاق'],
                "general_sound_quality": "صوت انسيابي مستمر",
            },
            "visual_form_semantics": ['خط مستقيم مع نقطة', 'مسار'],
            "core_semantic_axes": {
                "movement_progression": ('حركة/تقدم/زيادة', 'ثبات/نقصان'),
            },
            "general_connotations": ['الحركة', 'التقدم', 'الزيادة', 'الانزلاق'],
            "examples_from_basil": [
                "زاد، زيادة: تفيد النمو والزيادة",
                "زحف، انزلاق: تفيد الحركة الانسيابية",
                "زمن، زمان: تفيد الاستمرارية والتقدم",
            ],
            "notes_from_basil": "الزاي للحركة والتقدم والزيادة. النقطة فوق الحرف توحي بنقطة على مسار الحركة."
        },
        "س": {
            "character": "س",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "لثوي",
                "articulation_method": "احتكاكي مهموس",
                "sound_echoes": ['صوت الهمس', 'صوت الانسياب'],
                "general_sound_quality": "صوت انسيابي هامس",
            },
            "visual_form_semantics": ['خط متموج', 'مسار متعرج'],
            "core_semantic_axes": {
                "flow_continuity": ('انسياب/استمرار/سلاسة', 'توقف/تقطع'),
            },
            "general_connotations": ['الانسياب', 'الاستمرار', 'السلاسة', 'السير'],
            "examples_from_basil": [
                "سال، سيل: تفيد الانسياب والجريان",
                "سار، مسير: تفيد الحركة المستمرة",
                "سلس، سلاسة: تفيد السهولة والانسيابية",
            ],
            "notes_from_basil": "السين للانسياب والاستمرار والسلاسة. شكل الحرف المتموج يوحي بمسار انسيابي متعرج."
        },
        "ش": {
            "character": "ش",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "لثوي حنكي",
                "articulation_method": "احتكاكي مهموس",
                "sound_echoes": ['صوت التفشي', 'صوت الانتشار'],
                "general_sound_quality": "صوت منتشر متفشي",
            },
            "visual_form_semantics": ['خط متموج مع نقاط', 'انتشار وتفرع'],
            "core_semantic_axes": {
                "dispersion_branching": ('انتشار/تفرع/تشعب', 'تجمع/تركيز'),
            },
            "general_connotations": ['الانتشار', 'التفرع', 'التشعب', 'التفشي'],
            "examples_from_basil": [
                "شجرة: تفيد التفرع والانتشار",
                "شع، إشعاع: تفيد الانتشار من مركز",
                "شرح، شرح: تفيد التوسع والتفصيل",
            ],
            "notes_from_basil": "الشين للانتشار والتفرع والتشعب. النقاط الثلاث فوق الحرف توحي بالانتشار والتفرع من أصل واحد."
        },
        "ص": {
            "character": "ص",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "أسناني لثوي مفخم",
                "articulation_method": "احتكاكي مهموس مفخم",
                "sound_echoes": ['صوت الصلابة', 'صوت القوة'],
                "general_sound_quality": "صوت قوي مفخم",
            },
            "visual_form_semantics": ['دائرة مغلقة', 'وعاء محكم'],
            "core_semantic_axes": {
                "solidity_purity": ('صلابة/نقاء/خلوص', 'ليونة/تلوث'),
            },
            "general_connotations": ['الصلابة', 'النقاء', 'الخلوص', 'الإحكام'],
            "examples_from_basil": [
                "صلب، صلابة: تفيد القوة والمتانة",
                "صفا، صفاء: تفيد النقاء والخلوص",
                "صان، صيانة: تفيد الحفظ والحماية",
            ],
            "notes_from_basil": "الصاد للصلابة والنقاء والإحكام. شكل الحرف يوحي بدائرة مغلقة محكمة."
        },
        "ض": {
            "character": "ض",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "أسناني لثوي مفخم",
                "articulation_method": "انفجاري مجهور مفخم",
                "sound_echoes": ['صوت الضغط', 'صوت القوة'],
                "general_sound_quality": "صوت قوي مفخم ضاغط",
            },
            "visual_form_semantics": ['دائرة مغلقة مع نقطة', 'ضغط وقوة'],
            "core_semantic_axes": {
                "pressure_force": ('ضغط/قوة/إلزام', 'ضعف/تراخي'),
            },
            "general_connotations": ['الضغط', 'القوة', 'الإلزام', 'الضرورة'],
            "examples_from_basil": [
                "ضغط، ضاغط: تفيد الضغط والقوة",
                "ضرب، ضارب: تفيد التأثير القوي",
                "ضرورة، اضطرار: تفيد الإلزام والحتمية",
            ],
            "notes_from_basil": "الضاد للضغط والقوة والإلزام. النقطة فوق الحرف توحي بنقطة الضغط والتأثير."
        },
        "ط": {
            "character": "ط",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "أسناني لثوي مفخم",
                "articulation_method": "انفجاري مهموس مفخم",
                "sound_echoes": ['صوت الطرق', 'صوت الامتداد'],
                "general_sound_quality": "صوت قوي مفخم ممتد",
            },
            "visual_form_semantics": ['خط أفقي مع دائرة', 'امتداد وإحاطة'],
            "core_semantic_axes": {
                "extension_encirclement": ('امتداد/إحاطة/طول', 'قصر/محدودية'),
            },
            "general_connotations": ['الامتداد', 'الإحاطة', 'الطول', 'الشمول'],
            "examples_from_basil": [
                "طال، طويل: تفيد الامتداد والطول",
                "طاف، طواف: تفيد الدوران والإحاطة",
                "طبق، إطباق: تفيد الشمول والتغطية",
            ],
            "notes_from_basil": "الطاء للامتداد والإحاطة والشمول. شكل الحرف يوحي بامتداد أفقي مع إحاطة دائرية."
        },
        "ظ": {
            "character": "ظ",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "أسناني لثوي مفخم",
                "articulation_method": "احتكاكي مجهور مفخم",
                "sound_echoes": ['صوت الظهور', 'صوت البروز'],
                "general_sound_quality": "صوت قوي مفخم بارز",
            },
            "visual_form_semantics": ['خط أفقي مع دائرة ونقطة', 'ظهور وبروز'],
            "core_semantic_axes": {
                "appearance_prominence": ('ظهور/بروز/وضوح', 'خفاء/غموض'),
            },
            "general_connotations": ['الظهور', 'البروز', 'الوضوح', 'الظل'],
            "examples_from_basil": [
                "ظهر، ظهور: تفيد البروز والوضوح",
                "ظل، ظلال: تفيد الانعكاس والتجلي",
                "ظن، ظنون: تفيد التصور والتخيل",
            ],
            "notes_from_basil": "الظاء للظهور والبروز والوضوح. النقطة فوق الحرف توحي بنقطة الظهور والبروز."
        },
        "ع": {
            "character": "ع",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "حلقي",
                "articulation_method": "احتكاكي مجهور",
                "sound_echoes": ['صوت العمق', 'صوت الاتساع'],
                "general_sound_quality": "صوت عميق واسع",
            },
            "visual_form_semantics": ['عين مفتوحة', 'فتحة واسعة'],
            "core_semantic_axes": {
                "depth_knowledge": ('عمق/معرفة/إدراك', 'سطحية/جهل'),
                "width_comprehensiveness": ('اتساع/شمول/عموم', 'ضيق/خصوص'),
            },
            "general_connotations": ['العمق', 'المعرفة', 'الإدراك', 'الاتساع', 'الشمول', 'العموم'],
            "examples_from_basil": [
                "علم، معرفة: تفيد الإدراك والفهم",
                "عمق، عميق: تفيد البعد والغور",
                "عم، عموم: تفيد الشمول والاتساع",
            ],
            "notes_from_basil": "العين للعمق والمعرفة والاتساع. شكل الحرف يشبه العين المفتوحة التي ترى وتدرك، والفتحة الواسعة التي تشمل وتحيط."
        },
        "غ": {
            "character": "غ",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "حلقي",
                "articulation_method": "احتكاكي مجهور",
                "sound_echoes": ['صوت الغرغرة', 'صوت الغموض'],
                "general_sound_quality": "صوت عميق غامض",
            },
            "visual_form_semantics": ['عين مغلقة', 'غطاء'],
            "core_semantic_axes": {
                "mystery_covering": ('غموض/ستر/تغطية', 'وضوح/كشف'),
            },
            "general_connotations": ['الغموض', 'الستر', 'التغطية', 'الغياب'],
            "examples_from_basil": [
                "غطى، تغطية: تفيد الستر والإخفاء",
                "غاب، غياب: تفيد الاختفاء والبعد",
                "غمض، غموض: تفيد الإبهام وعدم الوضوح",
            ],
            "notes_from_basil": "الغين للغموض والستر والتغطية. النقطة فوق الحرف توحي بالعين المغلقة أو المغطاة."
        },
        "ف": {
            "character": "ف",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "شفوي أسناني",
                "articulation_method": "احتكاكي مهموس",
                "sound_echoes": ['صوت النفخ', 'صوت الفصل'],
                "general_sound_quality": "صوت هوائي فاصل",
            },
            "visual_form_semantics": ['فم مفتوح', 'فتحة'],
            "core_semantic_axes": {
                "separation_opening": ('فصل/فتح/فراغ', 'وصل/إغلاق/امتلاء'),
            },
            "general_connotations": ['الفصل', 'الفتح', 'الفراغ', 'الانفصال'],
            "examples_from_basil": [
                "فتح، فاتح: تفيد الفتح والكشف",
                "فصل، فاصل: تفيد القطع والتمييز",
                "فرغ، فراغ: تفيد الخلو والسعة",
            ],
            "notes_from_basil": "الفاء للفصل والفتح والفراغ. شكل الحرف يشبه الفم المفتوح أو الفتحة."
        },
        "ق": {
            "character": "ق",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "لهوي",
                "articulation_method": "انفجاري مهموس",
                "sound_echoes": ['صوت القطع', 'صوت القوة'],
                "general_sound_quality": "صوت قوي قاطع",
            },
            "visual_form_semantics": ['دائرة مع نقطتين', 'قوة وثبات'],
            "core_semantic_axes": {
                "strength_decisiveness": ('قوة/حسم/قطع', 'ضعف/تردد'),
            },
            "general_connotations": ['القوة', 'الحسم', 'القطع', 'الثبات'],
            "examples_from_basil": [
                "قطع، قاطع: تفيد الفصل الحاسم",
                "قوي، قوة: تفيد الشدة والمتانة",
                "قام، قيام: تفيد الثبات والاستقرار",
            ],
            "notes_from_basil": "القاف للقوة والحسم والقطع. النقطتان فوق الحرف توحيان بالثبات والقوة."
        },
        "ك": {
            "character": "ك",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "حنكي",
                "articulation_method": "انفجاري مهموس",
                "sound_echoes": ['صوت الكبت', 'صوت الكتم'],
                "general_sound_quality": "صوت مكتوم محبوس",
            },
            "visual_form_semantics": ['كف مقبوضة', 'إمساك'],
            "core_semantic_axes": {
                "restraint_possession": ('كبت/إمساك/احتواء', 'إطلاق/ترك'),
            },
            "general_connotations": ['الكبت', 'الإمساك', 'الاحتواء', 'التشبيه'],
            "examples_from_basil": [
                "كبت، كاتم: تفيد الحبس والمنع",
                "كف، كفاف: تفيد الإمساك والاحتواء",
                "كأن، مثل: تفيد التشبيه والمماثلة",
            ],
            "notes_from_basil": "الكاف للكبت والإمساك والاحتواء والتشبيه. شكل الحرف يشبه الكف المقبوضة التي تمسك شيئاً."
        },
        "ل": {
            "character": "ل",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "لثوي",
                "articulation_method": "جانبي",
                "sound_echoes": ['صوت اللين', 'صوت الانسياب'],
                "general_sound_quality": "صوت لين منساب",
            },
            "visual_form_semantics": ['خط منحني لأعلى', 'امتداد وارتفاع'],
            "core_semantic_axes": {
                "attachment_belonging": ('التصاق/انتماء/ملكية', 'انفصال/استقلال'),
            },
            "general_connotations": ['الالتصاق', 'الانتماء', 'الملكية', 'الاختصاص'],
            "examples_from_basil": [
                "لصق، التصاق: تفيد الارتباط والقرب",
                "له، لي: تفيد الملكية والاختصاص",
                "لأجل، لكي: تفيد التعليل والغاية",
            ],
            "notes_from_basil": "اللام للالتصاق والانتماء والملكية. شكل الحرف يوحي بالامتداد والارتفاع والاتصال."
        },
        "م": {
            "character": "م",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "شفوي",
                "articulation_method": "أنفي",
                "sound_echoes": ['صوت الضم', 'صوت الإغلاق'],
                "general_sound_quality": "صوت مغلق ممتلئ",
            },
            "visual_form_semantics": ['دائرة مغلقة', 'تجمع واكتمال'],
            "core_semantic_axes": {
                "completion_fullness": ('اكتمال/امتلاء/تمام', 'نقص/فراغ'),
            },
            "general_connotations": ['الاكتمال', 'الامتلاء', 'التمام', 'الجمع'],
            "examples_from_basil": [
                "تم، تمام: تفيد الاكتمال والنهاية",
                "جمع، مجموع: تفيد الضم والتجميع",
                "ملأ، امتلاء: تفيد الشغل والتعبئة",
            ],
            "notes_from_basil": "الميم للاكتمال والامتلاء والتمام. شكل الحرف يوحي بالدائرة المغلقة المكتملة."
        },
        "ن": {
            "character": "ن",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "لثوي",
                "articulation_method": "أنفي",
                "sound_echoes": ['صوت الأنين', 'صوت الرنين'],
                "general_sound_quality": "صوت رنان مستمر",
            },
            "visual_form_semantics": ['نقطة فوق حوض', 'بذرة في تربة'],
            "core_semantic_axes": {
                "emergence_growth": ('نمو/ظهور/بروز', 'كمون/خفاء'),
            },
            "general_connotations": ['النمو', 'الظهور', 'البروز', 'الاستمرار'],
            "examples_from_basil": [
                "نبت، نمو: تفيد الظهور والزيادة",
                "نور، إنارة: تفيد الإضاءة والوضوح",
                "نون: اسم الحرف يرتبط بالحوت والماء والحياة",
            ],
            "notes_from_basil": "النون للنمو والظهور والبروز. النقطة فوق الحرف توحي بالبذرة التي تنمو وتظهر."
        },
        "ه": {
            "character": "ه",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "حنجري",
                "articulation_method": "احتكاكي مهموس",
                "sound_echoes": ['صوت التنفس', 'صوت الهواء'],
                "general_sound_quality": "صوت هوائي خفيف",
            },
            "visual_form_semantics": ['دائرة مفتوحة', 'فراغ وهواء'],
            "core_semantic_axes": {
                "emptiness_lightness": ('فراغ/خفة/هواء', 'امتلاء/ثقل'),
            },
            "general_connotations": ['الفراغ', 'الخفة', 'الهواء', 'الهدوء'],
            "examples_from_basil": [
                "هواء، تهوية: تفيد الخفة والفراغ",
                "هدأ، هدوء: تفيد السكون والراحة",
                "هاء: اسم الحرف يرتبط بالتنفس والحياة",
            ],
            "notes_from_basil": "الهاء للفراغ والخفة والهواء. شكل الحرف يوحي بالدائرة المفتوحة أو الفراغ."
        },
        "و": {
            "character": "و",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "شفوي",
                "articulation_method": "شبه حركة",
                "sound_echoes": ['صوت الوصل', 'صوت الامتداد'],
                "general_sound_quality": "صوت ممتد موصول",
            },
            "visual_form_semantics": ['حلقة متصلة', 'وصلة'],
            "core_semantic_axes": {
                "connection_continuity": ('وصل/ربط/استمرار', 'فصل/قطع'),
            },
            "general_connotations": ['الوصل', 'الربط', 'الاستمرار', 'الجمع'],
            "examples_from_basil": [
                "وصل، واصل: تفيد الربط والاتصال",
                "ودام، دوام: تفيد الاستمرار والبقاء",
                "وجمع، مجموع: تفيد الضم والتجميع",
            ],
            "notes_from_basil": "الواو للوصل والربط والاستمرار. شكل الحرف يوحي بالحلقة المتصلة أو الوصلة."
        },
        "ي": {
            "character": "ي",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "حنكي",
                "articulation_method": "شبه حركة",
                "sound_echoes": ['صوت الامتداد', 'صوت الليونة'],
                "general_sound_quality": "صوت لين ممتد",
            },
            "visual_form_semantics": ['خط منحني مع نقطتين', 'يد ممدودة'],
            "core_semantic_axes": {
                "extension_possession": ('امتداد/ملكية/نسبة', 'انقطاع/انفصال'),
            },
            "general_connotations": ['الامتداد', 'الملكية', 'النسبة', 'الإضافة'],
            "examples_from_basil": [
                "يد، أيدي: تفيد الامتداد والقدرة",
                "لي، إليّ: تفيد الملكية والنسبة",
                "يمن، يمين: تفيد القوة والبركة",
            ],
            "notes_from_basil": "الياء للامتداد والملكية والنسبة. شكل الحرف يوحي باليد الممدودة أو الخط المنحني الممتد."
        },
        "ء": {
            "character": "ء",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "حنجري",
                "articulation_method": "انفجاري حنجري",
                "sound_echoes": ['صوت المفاجأة', 'صوت التوقف المفاجئ'],
                "general_sound_quality": "صوت مفاجئ قوي",
            },
            "visual_form_semantics": ['نقطة صغيرة', 'توقف مفاجئ'],
            "core_semantic_axes": {
                "surprise_interruption": ('مفاجأة/توقف/قطع', 'استمرار/تدفق'),
            },
            "general_connotations": ['المفاجأة', 'التوقف', 'القطع', 'البداية'],
            "examples_from_basil": [
                "سأل، مسألة: تفيد الاستفهام والمفاجأة",
                "بدأ، ابتداء: تفيد البدء والشروع",
                "قرأ، قراءة: تفيد النطق والتلفظ",
            ],
            "notes_from_basil": "الهمزة للمفاجأة والتوقف والقطع. شكل الحرف يوحي بالنقطة الصغيرة التي تمثل التوقف المفاجئ."
        },
        "ة": {
            "character": "ة",
            "language": "ar",
            "phonetic_properties": {
                "articulation_point": "حنجري",
                "articulation_method": "احتكاكي مهموس",
                "sound_echoes": ['صوت الهمس الخفيف', 'صوت النهاية'],
                "general_sound_quality": "صوت هامس خفيف",
            },
            "visual_form_semantics": ['وعاء صغير', 'احتواء وتأنيث'],
            "core_semantic_axes": {
                "femininity_containment": ('تأنيث/احتواء/نهاية', 'تذكير/انفتاح/استمرار'),
            },
            "general_connotations": ['التأنيث', 'الاحتواء', 'النهاية', 'التخصيص'],
            "examples_from_basil": [
                "معلمة، طالبة: تفيد التأنيث والتخصيص",
                "حديقة، غرفة: تفيد الاحتواء والإحاطة",
                "نهاية، خاتمة: تفيد الاكتمال والختام",
            ],
            "notes_from_basil": "التاء المربوطة للتأنيث والاحتواء والنهاية. شكل الحرف يوحي بالوعاء الصغير الذي يحتوي شيئاً ما."
        },
    },
    "en": {
        "A": {
            "character": "A",
            "language": "en",
            "phonetic_properties": {
                "sound_echoes": ["صوت الطفل 'آه' لطلب الحنان والقرب", 'صيحة تعجب أو إنذار'],
            },
            "visual_form_semantics": {'uppercase': ['قمة جبل', 'برج', 'رمز رأس الثور (Aleph)'], 'lowercase': ['أذن تستمع', 'ورقة شجر رقيقة', 'حلقة (مثل القرط)', 'خيمة أو مأوى']},
            "core_semantic_axes": {
                "authority_tenderness": ('جلال/طموح/سلطة/إعجاب/روعة (للشكل الكبير)', 'حنان/قرب/احتواء/دفء (للشكل الصغير)'),
            },
            "general_connotations": ['Apex', 'Ambition', 'Authority', 'Admiration', 'Awesome', 'affection', 'adore', 'accompany', 'arm', 'at home'],
            "examples_from_basil": [
            ],
            "notes_from_basil": "Does its triangular form not powerfully evoke the image of a mountain peak... suggesting strength and primacy? The open 'Aah' sound itself often forms the basis of exclamations... Does it not bring to mind the shape of an ear, listening intently? ...This feeling of containing, holding gently, seems present in words of closeness and care."
        },
        "B": {
            "character": "B",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "شفوي",
                "articulation_method": "انفجاري مجهور",
                "sound_echoes": ['صوت الانفجار الشفوي', 'صوت الامتلاء'],
            },
            "visual_form_semantics": {'uppercase': ['عمود مستقيم مع بروزين', 'جسم ممتلئ', 'حامل'], 'lowercase': ['بطن ممتلئة', 'حمل', 'احتواء']},
            "core_semantic_axes": {
                "fullness_birth": ('امتلاء/ولادة/بداية/بناء', 'فراغ/نهاية/هدم'),
            },
            "general_connotations": ['Birth', 'Begin', 'Build', 'Body', 'Bulge', 'baby', 'belly', 'bear', 'bulb', 'bubble'],
            "examples_from_basil": [
            ],
            "notes_from_basil": "The letter B, with its bulging forms, speaks of fullness, birth, and beginnings. Its very shape suggests pregnancy, a belly swollen with new life. The sound itself requires the lips to close and then burst open - like the moment of birth."
        },
        "O": {
            "character": "O",
            "language": "en",
            "phonetic_properties": {
                "sound_echoes": ['صوت الدهشة والتعجب', 'صوت الدائرة المكتملة'],
            },
            "visual_form_semantics": {'uppercase': ['دائرة كاملة', 'عين مفتوحة', 'فم متعجب'], 'lowercase': ['دائرة صغيرة', 'نقطة', 'بذرة']},
            "core_semantic_axes": {
                "wholeness_emptiness": ('اكتمال/شمول/دائرية', 'فراغ/خواء/انفتاح'),
            },
            "general_connotations": ['Orb', 'Omnipresent', 'Overall', 'Observe', 'Open', 'origin', 'orbit', 'only', 'omen', 'odd'],
            "examples_from_basil": [
            ],
            "notes_from_basil": "O is the perfect circle, the shape of completion and wholeness. It is the shape of the eye that observes, the mouth that exclaims in wonder, the cycle that returns to its beginning. The sound itself requires the mouth to form a perfect circle - a physical embodiment of its meaning."
        },
        "E": {
            "character": "E",
            "language": "en",
            "phonetic_properties": {
                "sound_echoes": ['صوت الانفتاح المتوسط', 'صوت الامتداد الأفقي'],
            },
            "visual_form_semantics": {'uppercase': ['رفوف متعددة', 'سلم أفقي', 'تقسيمات منظمة'], 'lowercase': ['نصف دائرة صغيرة', 'فتحة جزئية']},
            "core_semantic_axes": {
                "extension_organization": ('امتداد/تنظيم/تقسيم', 'تركيز/تجميع'),
            },
            "general_connotations": ['Extend', 'Equal', 'Establish', 'Enumerate', 'Educate', 'edge', 'emerge', 'emit', 'echo', 'end'],
            "examples_from_basil": [
            ],
            "notes_from_basil": "E with its multiple horizontal lines suggests organization, extension, and equality. It is the sound of emergence and expression. The lowercase 'e' appears as a half-circle, suggesting partial enclosure or the beginning of emergence."
        },
        "I": {
            "character": "I",
            "language": "en",
            "phonetic_properties": {
                "sound_echoes": ['صوت حاد رفيع', 'صوت الأنا والذات'],
            },
            "visual_form_semantics": {'uppercase': ['عمود مستقيم', 'شخص واقف', 'الذات المنفردة'], 'lowercase': ['نقطة مع خط قصير', 'شخص صغير', 'ذات محدودة']},
            "core_semantic_axes": {
                "identity_individuality": ('هوية/فردية/ذاتية', 'جماعية/غيرية'),
            },
            "general_connotations": ['Identity', 'Individual', 'Intense', 'Immediate', 'Inspire', 'intimate', 'inner', 'idea', 'imagine', 'illuminate'],
            "examples_from_basil": [
            ],
            "notes_from_basil": "I stands tall and alone - the perfect symbol of individual identity. It is the self, standing upright. The sound is high and sharp, cutting through space like a beam of light. The lowercase 'i' with its dot suggests a small person, a single point of consciousness."
        },
        "N": {
            "character": "N",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "لثوي",
                "articulation_method": "أنفي",
                "sound_echoes": ['صوت الرنين الأنفي', 'صوت النفي'],
            },
            "visual_form_semantics": {'uppercase': ['جسر', 'رابط بين عمودين', 'حركة صعود وهبوط'], 'lowercase': ['قوس صغير مع عمود', 'حركة انحناء']},
            "core_semantic_axes": {
                "connection_negation": ('ربط/اتصال/شبكة', 'نفي/رفض/عكس'),
            },
            "general_connotations": ['Network', 'Navigate', 'Nexus', 'Negotiate', 'Nurture', 'near', 'node', 'nest', 'narrow', 'no'],
            "examples_from_basil": [
            ],
            "notes_from_basil": "N bridges two vertical pillars - connecting separate entities. It suggests movement, navigation between points. The sound resonates in the nose, creating a humming connection. It also forms the basis of negation in many languages - 'no', 'not', 'never' - reversing or connecting opposites."
        },
        "M": {
            "character": "M",
            "language": "en",
            "phonetic_properties": {
                "articulation_point": "شفوي",
                "articulation_method": "أنفي",
                "sound_echoes": ['صوت الأم', 'صوت الرنين الشفوي'],
            },
            "visual_form_semantics": {'uppercase': ['جبال متعددة', 'موجات', 'ثديان'], 'lowercase': ['قوسان متصلان', 'موجة صغيرة']},
            "core_semantic_axes": {
                "motherhood_magnitude": ('أمومة/عظمة/كثرة', 'فردية/صغر'),
            },
            "general_connotations": ['Mother', 'Magnitude', 'Multiple', 'Mountain', 'Massive', 'many', 'more', 'merge', 'mutual', 'mild'],
            "examples_from_basil": [
            ],
            "notes_from_basil": "M is the mother letter, formed by pressing the lips together as a baby does when nursing. Its shape suggests mountains, waves, or breasts - symbols of nurturing abundance. The sound is the first many babies make - 'mama' - and resonates with comfort and sustenance."
        },
        "W": {
            "character": "W",
            "language": "en",
            "phonetic_properties": {
                "sound_echoes": ['صوت الدهشة والتعجب', 'صوت الحركة المتموجة'],
            },
            "visual_form_semantics": {'uppercase': ['موجات متعددة', 'وديان', 'حركة متذبذبة'], 'lowercase': ['موجات صغيرة', 'تموجات متتالية']},
            "core_semantic_axes": {
                "wave_wonder": ('تموج/تذبذب/حركة', 'ثبات/استقرار'),
            },
            "general_connotations": ['Wave', 'Wonder', 'Wander', 'Water', 'Wind', 'wiggle', 'weave', 'womb', 'well', 'way'],
            "examples_from_basil": [
            ],
            "notes_from_basil": "W is water in motion - waves rising and falling. Its shape perfectly captures the undulating movement of water, wind, or wandering paths. The sound itself requires the lips to form a small circle that opens and closes - like the motion of waves."
        },
        "Y": {
            "character": "Y",
            "language": "en",
            "phonetic_properties": {
                "sound_echoes": ['صوت التساؤل', 'صوت التفرع'],
            },
            "visual_form_semantics": {'uppercase': ['طريق متفرع', 'غصن شجرة', 'خيارات متعددة'], 'lowercase': ['خط منحني مع ذيل', 'بذرة نابتة']},
            "core_semantic_axes": {
                "choice_yearning": ('اختيار/تفرع/تساؤل', 'حسم/تحديد'),
            },
            "general_connotations": ['Yes', 'Yield', 'Yearning', 'Youth', 'Yoke', 'young', 'yield', 'yearn', 'yawn', 'yell'],
            "examples_from_basil": [
            ],
            "notes_from_basil": "Y stands at the crossroads of choice - a single path that divides. It is the question that seeks answers, the yearning that reaches upward. The sound itself is formed by the tongue reaching up in the mouth, striving toward something higher."
        },
        "Z": {
            "character": "Z",
            "language": "en",
            "phonetic_properties": {
                "sound_echoes": ['صوت الحركة السريعة', 'صوت القطع'],
            },
            "visual_form_semantics": {'uppercase': ['برق', 'مسار متعرج', 'سيف'], 'lowercase': ['خط متعرج صغير', 'حركة سريعة']},
            "core_semantic_axes": {
                "speed_finality": ('سرعة/حدة/نهاية', 'بطء/بداية'),
            },
            "general_connotations": ['Zap', 'Zigzag', 'Zealous', 'Zero', 'Zenith', 'zip', 'zoom', 'zest', 'zeal', 'zone'],
            "examples_from_basil": [
            ],
            "notes_from_basil": "Z cuts through space like lightning - quick, sharp, and final. It is the last letter, the end of the sequence, yet full of energy and movement. The sound itself is like a blade cutting air - swift and decisive."
        },
    }
}
