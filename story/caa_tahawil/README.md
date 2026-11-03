# قاع التهاويل: الصوت الذي يعرف اسمك

مشروع إنتاج صوتي مستقل داخل مستودع MPC، لا يغيّر أو يستدعي ملفات دبلجة الدراما الصينية الحالية.

## محتويات المشروع
- story_segments.json: النص مقسّم إلى 41 مقطع أداء، وستة فصول، مع علامات أداء ووقفات.
- story_full_text.md: النص المقروء كاملاً دون علامات TTS.
- render_horror_story.py: توليد قابل للاستئناف باستخدام Gemini 3.8 Flash TTS، مع حفظ كل WAV جزئي.
- mix_spec.json: مواصفات المكساج والمؤثرات المستمدة من التحليل.

## إعداد الصوت
- نموذج: gemini-3.8-flash-tts عبر Gemini Interactions API.
- الصوت: Fenrir، صوت واحد ثابت للراوي.
- الأداء: باريتون رجولي دافئ، سرد خليجي/سعودي مفهوم، وتغيّر الإيقاع والمشاعر عبر speech_metadata.style.
- لا يجري استنساخ أو تقليد هوية الراوي البيومترية في التسجيلات المرجعية؛ المرجع يحدد صفات أداء عامة فقط.

## التشغيل
GitHub Actions: Arabic horror narration – Qaa Al-Tahawil عبر workflow_dispatch في .github/workflows/arabic-horror-narration.yml.
يتطلب السر GOOGLE_AI (أو GOOGLE_AI_PAID). الناتج النهائي:
- FLAC ستيريو 48 kHz lossless.
- MP3 ستيريو 48 kHz بسرعة 320 kbps.
- WAV سرد نظيف 24 kHz mono.
- manifest وقياسات مدة كل فصل.

تُحفظ المقاطع الجزئية في cache حتى يمكن استئناف التشغيل بعد انقطاعه.
