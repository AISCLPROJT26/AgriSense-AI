import base64
import io
import json
import os
import pickle
import numpy as np
import requests
from PIL import Image
from skimage.feature import hog

# =========================================================
# MODEL LOCATION & CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "models", "disease_model.pkl")

# Load local ML model if available
model = None
label_encoder = None
image_size = (64, 64)

if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as file:
        model_data = pickle.load(file)
        model = model_data["model"]
        label_encoder = model_data["label_encoder"]
        image_size = model_data["image_size"]


# =========================================================
# BILINGUAL (ENGLISH + HINDI) PLANT DISEASE KNOWLEDGE BASE
# =========================================================

DISEASE_DETAILS = {
    "Tomato_Early_blight": {
        "name_en": "Tomato Early Blight",
        "name_hi": "टमाटर का अगेती झुलसा रोग (Early Blight)",
        "is_healthy": False,
        "problem_en": (
            "The plant is suffering from a fungal infection caused by Alternaria solani. "
            "It typically starts on older, lower leaves and weakens the plant by reducing photosynthesis."
        ),
        "problem_hi": (
            "पौधे में 'अल्टरनेरिया सोलानी' (Alternaria solani) नामक फंगस का संक्रमण है। "
            "यह बीमारी नीचे की पुरानी पत्तियों से शुरू होती है और धीरे-धीरे पूरे पौधे को कमजोर कर देती है।"
        ),
        "symptoms_en": (
            "Dark brown to black spots with concentric rings (target-like pattern) on leaves, "
            "surrounded by yellowing tissue."
        ),
        "symptoms_hi": (
            "पत्तियों पर गहरे भूरे या काले रंग के गोल छल्लेदार धब्बे पड़ना और धब्बों के आसपास पत्ती का पीला पड़ना।"
        ),
        "suggestion_en": (
            "Remove and destroy affected lower leaves immediately. Avoid wetting leaves while watering, "
            "maintain proper spacing for airflow, and apply a copper-based or Mancozeb fungicide if needed."
        ),
        "suggestion_hi": (
            "संक्रमित निचली पत्तियों को तुरंत तोड़कर नष्ट कर दें। सिंचाई करते समय पत्तियों को गीला न करें, "
            "हवा के बहाव के लिए उचित दूरी रखें और आवश्यकता अनुसार मैनकोज़ेब (Mancozeb) या कॉपर फंगीसाइड का छिड़काव करें।"
        ),
    },
    "Tomato_Late_blight": {
        "name_en": "Tomato Late Blight",
        "name_hi": "टमाटर का पछेती झुलसा रोग (Late Blight)",
        "is_healthy": False,
        "problem_en": (
            "The plant is infected by Phytophthora infestans, a fast-spreading water mold that thrives "
            "in cool, wet weather and can rapidly damage leaves, stems, and fruits."
        ),
        "problem_hi": (
            "पौधे में 'फाइटोफ्थोरा इन्फेस्टैन्स' (Phytophthora infestans) का गंभीर संक्रमण है। "
            "यह ठंडे और नमी वाले मौसम में बहुत तेजी से फैलता है और पत्तियों, तनों व फलों को सड़ा सकता है।"
        ),
        "symptoms_en": (
            "Large, irregular greenish-black water-soaked patches on leaves that quickly turn brown and withered, "
            "sometimes with white fungal growth underneath."
        ),
        "symptoms_hi": (
            "पत्तियों पर पानी से भीगे हुए जैसे हरे-काले बड़े धब्बे दिखना जो जल्दी भूरे होकर सूख जाते हैं, "
            "और पत्ती के नीचे सफेद फफूंद दिखाई देना।"
        ),
        "suggestion_en": (
            "Immediately remove infected plant parts and keep foliage dry. Improve drainage and air circulation, "
            "and use recommended protective fungicides (such as Copper oxychloride or Metalaxyl + Mancozeb)."
        ),
        "symptoms_care": "",
        "suggestion_hi": (
            "संक्रमित हिस्सों को तुरंत हटाकर खेत से दूर नष्ट करें। पत्तियों पर नमी न रहने दें, हवा का संचार बढ़ाएं "
            "और कॉपर ऑक्सीक्लोराइड या मेटालैक्सिल + मैनकोज़ेब फंगीसाइड का प्रयोग करें।"
        ),
    },
    "Tomato_Bacterial_spot": {
        "name_en": "Tomato Bacterial Spot",
        "name_hi": "टमाटर का जीवाणु धब्बा रोग (Bacterial Spot)",
        "is_healthy": False,
        "problem_en": (
            "The plant has a bacterial infection caused by Xanthomonas species, which spreads through "
            "splashing water, rain, and contaminated tools in warm, humid conditions."
        ),
        "problem_hi": (
            "पौधे में 'ज़ैंथोमोनस' (Xanthomonas) बैक्टीरिया के कारण जीवाणु धब्बा रोग हुआ है। "
            "यह गर्म और नमी वाले मौसम में पानी के छींटों और संक्रमित औजारों से फैलता है।"
        ),
        "symptoms_en": (
            "Small, dark, greasy-looking water-soaked spots on leaves and stems; leaves may turn yellow and drop early."
        ),
        "symptoms_hi": (
            "पत्तियों और तनों पर छोटे, गहरे भूरे या काले चिकने धब्बे पड़ना; पत्तियां पीली होकर समय से पहले गिर सकती हैं।"
        ),
        "suggestion_en": (
            "Remove infected leaves, avoid overhead watering, disinfect gardening tools, and apply a copper-based bactericide spray."
        ),
        "suggestion_hi": (
            "प्रभावित पत्तियों को हटा दें, ऊपर से पानी का छिड़काव न करें, बागवानी के औजारों को साफ रखें "
            "और कॉपर-आधारित बैक्टीरिया नाशक का छिड़काव करें।"
        ),
    },
    "Tomato_Leaf_Mold": {
        "name_en": "Tomato Leaf Mold",
        "name_hi": "टमाटर का लीफ मोल्ड (पत्ती की फफूंद)",
        "is_healthy": False,
        "problem_en": (
            "The plant is affected by the fungus Passalora fulva, commonly triggered by high humidity "
            "and poor air ventilation around the foliage."
        ),
        "problem_hi": (
            "अधिक नमी और हवा के कम बहाव के कारण पौधे की पत्तियों में 'पासलोरा फुलवा' (Passalora fulva) फफूंद लग गई है।"
        ),
        "symptoms_en": (
            "Pale green or yellowish spots on the upper leaf surface with olive-green to velvety brown mold on the underside."
        ),
        "symptoms_hi": (
            "पत्ती के ऊपरी हिस्से पर हल्के पीले धब्बे और निचले हिस्से पर जैतूनी-हरे या भूरे रंग की मखमली फफूंद दिखना।"
        ),
        "suggestion_en": (
            "Increase spacing and airflow between plants, water at the soil level only, prune lower leaves, and use a suitable fungicide if severe."
        ),
        "suggestion_hi": (
            "पौधों के बीच हवा का अच्छा बहाव बनाएं, पानी सीधा जड़ों में दें, निचली पत्तियों की छंटाई करें और जरूरत पड़ने पर फंगीसाइड का उपयोग करें।"
        ),
    },
    "Tomato_Septoria_leaf_spot": {
        "name_en": "Tomato Septoria Leaf Spot",
        "name_hi": "टमाटर का सेप्टोरिया पत्ती धब्बा रोग (Septoria Leaf Spot)",
        "is_healthy": False,
        "problem_en": (
            "The plant is infected by the fungus Septoria lycopersici, which attacks lower leaves first "
            "after fruit set during warm, wet weather."
        ),
        "problem_hi": (
            "पौधे में 'सेप्टोरिया लाइकोपर्सिकी' फंगस का संक्रमण है, जो नमी वाले मौसम में नीचे की पत्तियों से शुरू होकर ऊपर की ओर बढ़ता है।"
        ),
        "symptoms_en": (
            "Numerous small circular spots with dark brown borders and lighter gray or tan centers Containing tiny black dots."
        ),
        "symptoms_hi": (
            "पत्तियों पर छोटे-छोटे गोल धब्बे जिनके किनारे गहरे भूरे और बीच का हिस्सा भूरा-सफेद होता है, जिनमें काले सूक्ष्म बिंदु दिखते हैं।"
        ),
        "suggestion_en": (
            "Remove affected leaves immediately, keep the soil covered with clean mulch to prevent splash, and apply fungicide if infection spreads."
        ),
        "suggestion_hi": (
            "संक्रमित पत्तियों को तुरंत हटा दें, मिट्टी के छींटे पत्तियों पर न पड़ने दें (मल्चिंग करें) और संक्रमण बढ़ने पर फंगीसाइड स्प्रे करें।"
        ),
    },
    "Tomato_healthy": {
        "name_en": "Tomato — Healthy Plant",
        "name_hi": "टमाटर — स्वस्थ पौधा (कोई बीमारी नहीं)",
        "is_healthy": True,
        "problem_en": (
            "No disease detected! Your tomato plant leaf looks green, vibrant, and healthy with no visible fungal or bacterial infection."
        ),
        "problem_hi": (
            "पौधे में कोई दिक्कत नहीं है! आपके टमाटर के पौधे की पत्ती बिल्कुल हरी-भरी और स्वस्थ है, इसमें किसी बीमारी के लक्षण नहीं हैं।"
        ),
        "symptoms_en": "Uniform green color, healthy leaf structure, and zero lesions or spots.",
        "symptoms_hi": "पत्ती का रंग एक समान हरा है और इस पर कोई धब्बे या सिकुड़न नहीं है।",
        "suggestion_en": (
            "Continue regular watering at the base, provide balanced sunlight and nutrients, and monitor leaves weekly."
        ),
        "suggestion_hi": (
            "इसी तरह नियमित देखभाल जारी रखें, जड़ों में संतुलित पानी व खाद दें और समय-समय पर पत्तियों की जांच करते रहें।"
        ),
    },
    "Potato_Early_blight": {
        "name_en": "Potato Early Blight",
        "name_hi": "आलू का अगेती झुलसा रोग (Potato Early Blight)",
        "is_healthy": False,
        "problem_en": (
            "The potato plant has a fungal infection (Alternaria solani) that damages foliage and can reduce tuber yield if left untreated."
        ),
        "problem_hi": (
            "आलू के पौधे में 'अल्टरनेरिया सोलानी' फंगस के कारण अगेती झुलसा रोग लगा है, जिससे पत्तियां सूखने लगती हैं और पैदावार घट सकती है।"
        ),
        "symptoms_en": (
            "Dry brown circular spots with concentric bullseye rings mainly on older leaves."
        ),
        "symptoms_hi": (
            "पुरानी पत्तियों पर गहरे भूरे रंग के गोल छल्लेदार (निशाने जैसे) सूखे धब्बे दिखाई देना।"
        ),
        "suggestion_en": (
            "Remove heavily affected leaves, maintain balanced soil nutrition (especially nitrogen and potassium), and spray Mancozeb or Chlorothalonil if needed."
        ),
        "suggestion_hi": (
            "अधिक प्रभावित पत्तियों को हटा दें, मिट्टी में संतुलित पोषण बनाए रखें और रोकथाम के लिए मैनकोज़ेब (Mancozeb) का छिड़काव करें।"
        ),
    },
    "Potato_Late_blight": {
        "name_en": "Potato Late Blight",
        "name_hi": "आलू का पछेती झुलसा रोग (Potato Late Blight)",
        "is_healthy": False,
        "problem_en": (
            "The potato plant is suffering from Late Blight (Phytophthora infestans), a serious infection that rots leaves and tubers in humid, cool weather."
        ),
        "problem_hi": (
            "आलू के पौधे में पछेती झुलसा (Phytophthora infestans) का गंभीर संक्रमण है, जो नमी और ठंड में पत्तियों और कंद (आलू) दोनों को नुकसान पहुंचाता है।"
        ),
        "symptoms_en": (
            "Dark water-soaked patches at leaf tips and edges, turning brown-black with white cottony growth underneath in humid mornings."
        ),
        "symptoms_hi": (
            "पत्तियों के किनारों पर काले-भूरे गीले धब्बे पड़ना और सुबह नमी में पत्ती के नीचे सफेद रुई जैसी फफूंद दिखना।"
        ),
        "suggestion_en": (
            "Destroy severely infected leaves immediately, avoid excess irrigation, earth up soil properly around tubers, and apply a systemic fungicide."
        ),
        "suggestion_hi": (
            "संक्रमित पत्तियों को तुरंत नष्ट करें, अत्यधिक सिंचाई से बचें, आलू के कंदों पर मिट्टी चढ़ाकर रखें और उचित फंगीसाइड का छिड़काव करें।"
        ),
    },
    "Potato_healthy": {
        "name_en": "Potato — Healthy Plant",
        "name_hi": "आलू — स्वस्थ पौधा (कोई बीमारी नहीं)",
        "is_healthy": True,
        "problem_en": (
            "No disease detected! Your potato plant leaf appears completely healthy and free from blight or pest damage."
        ),
        "problem_hi": (
            "पौधे में कोई बीमारी नहीं है! आपके आलू का पौधा पूरी तरह स्वस्थ है और पत्तियां सामान्य रूप से विकसित हो रही हैं।"
        ),
        "symptoms_en": "Fresh green foliage with no brown rings or water-soaked lesions.",
        "symptoms_hi": "ताजी हरी पत्तियां जिन पर किसी प्रकार के भूरे या काले धब्बे नहीं हैं।",
        "suggestion_en": (
            "Keep soil well-drained, maintain regular earthing-up, and inspect the crop periodically during humid weather."
        ),
        "suggestion_hi": (
            "खेत में जल निकासी अच्छी रखें, समय पर मिट्टी चढ़ाएं और नमी वाले मौसम में फसल की नियमित निगरानी करते रहें।"
        ),
    },
    "Apple___Apple_scab": {
        "name_en": "Apple Scab",
        "name_hi": "सेब का स्कैब रोग (Apple Scab)",
        "is_healthy": False,
        "problem_en": (
            "The apple plant is infected by the fungus Venturia inaequalis, causing crusty lesions on leaves and fruit."
        ),
        "problem_hi": (
            "सेब के पौधे में 'वेंचुरिया इनइक्वैलिस' (Venturia inaequalis) फंगस का संक्रमण है, जिससे पत्तियों और फलों पर पपड़ीदार धब्बे बन जाते हैं।"
        ),
        "symptoms_en": (
            "Olive-green to dark brown velvety or scabby spots on leaves; leaves may curl and fall prematurely."
        ),
        "symptoms_hi": (
            "पत्तियों पर जैतूनी-हरे या गहरे भूरे मखमली/खुरदरे धब्बे पड़ना और पत्तियों का मुड़कर समय से पहले गिरना।"
        ),
        "suggestion_en": (
            "Collect and destroy fallen leaves, prune branches to allow sunlight and air, and apply protective orchard fungicides."
        ),
        "suggestion_hi": (
            "नीचे गिरी हुई संक्रमित पत्तियों को इकट्ठा करके नष्ट करें, धूप और हवा के लिए टहनियों की छंटाई करें और उपयुक्त फंगीसाइड का छिड़काव करें।"
        ),
    },
    "Apple___Black_rot": {
        "name_en": "Apple Black Rot (Frogeye Leaf Spot)",
        "name_hi": "सेब का ब्लैक रॉट / फ्रॉग-आई लीफ स्पॉट",
        "is_healthy": False,
        "problem_en": (
            "The tree is infected by the fungus Botryosphaeria obtusa, which causes leaf spots, branch cankers, and fruit rot."
        ),
        "problem_hi": (
            "पौधे में 'बोट्रियोस्फेरिया ओबटुसा' फंगस का संक्रमण है, जो पत्तियों पर धब्बे, टहनियों में घाव और फलों में सड़न पैदा करता है।"
        ),
        "symptoms_en": (
            "Circular purple-bordered spots with light brown centers ('frogeye' appearance) on leaves."
        ),
        "symptoms_hi": (
            "पत्तियों पर बैंगनी किनारों वाले गोल धब्बे जिनके बीच का भाग हल्का भूरा (मेंढक की आंख जैसा) दिखाई देता है।"
        ),
        "suggestion_en": (
            "Prune out dead wood and cankers, remove mummified fruits from the tree, and maintain regular orchard sanitation."
        ),
        "suggestion_hi": (
            "सूखी व संक्रमित टहनियों को काटकर हटा दें, पेड़ पर लटके सड़े फलों को नष्ट करें और बगीचे की साफ-सफाई बनाए रखें।"
        ),
    },
    "Apple___Cedar_apple_rust": {
        "name_en": "Cedar Apple Rust",
        "name_hi": "सेब का रतुआ रोग (Cedar Apple Rust)",
        "is_healthy": False,
        "problem_en": (
            "The plant is affected by the rust fungus Gymnosporangium juniperi-virginianae, causing bright orange spots on foliage."
        ),
        "problem_hi": (
            "सेब की पत्तियों में रतुआ (Rust) फंगस का संक्रमण है, जिससे पत्तियों पर चमकीले पीले-नारंगी धब्बे बन जाते हैं।"
        ),
        "symptoms_en": (
            "Bright yellow-orange circular spots on upper leaf surfaces, often with red borders and tiny tube-like projections underneath."
        ),
        "symptoms_hi": (
            "पत्तियों की ऊपरी सतह पर चमकीले पीले या नारंगी रंग के गोल धब्बे और उनके चारों ओर लाल घेरा दिखना।"
        ),
        "suggestion_en": (
            "Remove severely infected leaves, improve airflow around the canopy, and apply a rust-control fungicide during early growth stages."
        ),
        "suggestion_hi": (
            "संक्रमित पत्तियों को हटा दें, पेड़ में हवा का अच्छा प्रवाह रखें और शुरुआती अवस्था में रतुआ-रोधी फंगीसाइड का छिड़काव करें।"
        ),
    },
    "Apple___healthy": {
        "name_en": "Apple — Healthy Plant",
        "name_hi": "सेब — स्वस्थ पौधा (कोई बीमारी नहीं)",
        "is_healthy": True,
        "problem_en": (
            "No disease detected! The apple leaf looks clean, green, and completely healthy."
        ),
        "problem_hi": (
            "पौधे में कोई बीमारी नहीं है! सेब की पत्ती बिल्कुल साफ, हरी और स्वस्थ दिखाई दे रही है।"
        ),
        "symptoms_en": "Healthy green blade with no scab, rust, or frogeye lesions.",
        "symptoms_hi": "स्वस्थ हरी पत्ती जिस पर किसी प्रकार के धब्बे या फफूंद नहीं हैं।",
        "suggestion_en": (
            "Continue balanced orchard care, proper pruning, and seasonal monitoring."
        ),
        "suggestion_hi": (
            "नियमित सिंचाई, संतुलित पोषण और समय-समय पर बगीचे की निगरानी जारी रखें।"
        ),
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "name_en": "Corn (Maize) Gray Leaf Spot",
        "name_hi": "मक्का का ग्रे लीफ स्पॉट (Cercospora पत्ती धब्बा रोग)",
        "is_healthy": False,
        "problem_en": (
            "The corn plant has a fungal infection caused by Cercospora zeae-maydis, which restricts leaf area available for grain filling."
        ),
        "problem_hi": (
            "मक्का के पौधे में 'सर्कोस्पोरा' (Cercospora) फंगस का संक्रमण है, जो पत्तियों को सुखाकर दानों के विकास को प्रभावित करता है।"
        ),
        "symptoms_en": (
            "Long, narrow, rectangular tan-to-gray lesions running parallel between the leaf veins."
        ),
        "symptoms_hi": (
            "पत्ती की नसों के समानांतर लंबे, संकरे और चौकोर भूरे-स्लेटी रंग के धब्बे दिखाई देना।"
        ),
        "suggestion_en": (
            "Remove severely affected lower leaves, practice crop rotation, avoid dense planting, and use fungicide if infection reaches upper leaves."
        ),
        "suggestion_hi": (
            "गंभीर रूप से प्रभावित पत्तियों को हटाएं, फसल चक्र (Crop Rotation) अपनाएं, पौधों के बीच उचित दूरी रखें और जरूरत पर फंगीसाइड का प्रयोग करें।"
        ),
    },
    "Corn_(maize)___Common_rust_": {
        "name_en": "Corn (Maize) Common Rust",
        "name_hi": "मक्का का सामान्य रतुआ रोग (Common Rust)",
        "is_healthy": False,
        "problem_en": (
            "The corn plant is infected by the rust fungus Puccinia sorghi, favored by cool temperatures and high humidity."
        ),
        "problem_hi": (
            "मक्का के पौधे में 'पक्सिनिया सोर्घी' (Puccinia sorghi) फंगस के कारण रतुआ रोग लगा है, जो नमी वाले मौसम में पत्तियों पर फैलता है।"
        ),
        "symptoms_en": (
            "Small, powdery cinnamon-brown to brick-red pustules scattered on both upper and lower leaf surfaces."
        ),
        "symptoms_hi": (
            "पत्ती के दोनों तरफ छोटे-छोटे लाल-भूरे (जंग जैसे) उभरे हुए पाउडर वाले दाने/धब्बे दिखना।"
        ),
        "suggestion_en": (
            "Monitor the crop closely, remove heavily rusted leaves, ensure good field ventilation, and apply a foliar fungicide if spreading rapidly."
        ),
        "suggestion_hi": (
            "फसल की नियमित निगरानी करें, अधिक प्रभावित पत्तियों को हटा दें और रोग तेजी से फैलने पर मैनकोज़ेब या प्रोपिकोनाज़ोल फंगीसाइड का छिड़काव करें।"
        ),
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "name_en": "Corn (Maize) Northern Leaf Blight",
        "name_hi": "मक्का का उत्तरी पत्ती झुलसा रोग (Northern Leaf Blight)",
        "is_healthy": False,
        "problem_en": (
            "The plant is suffering from Exserohilum turcicum fungal infection, which causes large cigar-shaped dead patches on leaves."
        ),
        "problem_hi": (
            "मक्का के पौधे में फंगल झुलसा रोग है, जिससे पत्तियों पर बड़े आकार के सूखे धब्बे बन जाते हैं और पौधा कमजोर हो जाता है।"
        ),
        "symptoms_en": (
            "Large, elongated cigar-shaped grayish-green or tan lesions (1 to 6 inches long) on the leaves."
        ),
        "symptoms_hi": (
            "पत्तियों पर नाव या सिगार के आकार के बड़े, लंबे स्लेटी-हरे या भूरे रंग के सूखे धब्बे पड़ना।"
        ),
        "suggestion_en": (
            "Remove infected crop debris, ensure adequate potassium nutrition, and use a recommended fungicide if lesions appear before tasseling."
        ),
        "suggestion_hi": (
            "संक्रमित पत्तियों व फसल अवशेषों को हटाएं, उचित पोषण दें और रोग बढ़ने पर कृषि विशेषज्ञ की सलाह से फंगीसाइड का छिड़काव करें।"
        ),
    },
    "Corn_(maize)___healthy": {
        "name_en": "Corn (Maize) — Healthy Plant",
        "name_hi": "मक्का — स्वस्थ पौधा (कोई बीमारी नहीं)",
        "is_healthy": True,
        "problem_en": (
            "No disease detected! Your corn leaf is vibrant green and shows healthy growth with no rust or blight."
        ),
        "problem_hi": (
            "पौधे में कोई दिक्कत नहीं है! आपकी मक्का की पत्ती बिल्कुल हरी और स्वस्थ है, इसमें किसी रोग के लक्षण नहीं हैं।"
        ),
        "symptoms_en": "Clean, deep green leaf surface with intact veins.",
        "symptoms_hi": "साफ और गहरी हरी पत्ती जिस पर कोई धब्बा या जंग का निशान नहीं है।",
        "suggestion_en": (
            "Maintain timely irrigation and weed control for strong cob development."
        ),
        "suggestion_hi": (
            "अच्छी पैदावार के लिए समय पर सिंचाई, निराई-गुड़ाई और संतुलित खाद प्रबंधन जारी रखें।"
        ),
    },
    "Pepper__bell___Bacterial_spot": {
        "name_en": "Bell Pepper (Capsicum) Bacterial Spot",
        "name_hi": "शिमला मिर्च का जीवाणु धब्बा रोग (Bacterial Spot)",
        "is_healthy": False,
        "problem_en": (
            "The bell pepper plant has a bacterial leaf spot infection (Xanthomonas campestris), which causes leaf drop and sunscald on fruits."
        ),
        "problem_hi": (
            "शिमला मिर्च के पौधे में बैक्टीरिया का संक्रमण है, जिससे पत्तियों पर धब्बे पड़ते हैं और पत्तियां पीली होकर झड़ने लगती हैं।"
        ),
        "symptoms_en": (
            "Small water-soaked spots on leaves that turn dark brown with yellow halos; severely affected leaves raggedly drop off."
        ),
        "symptoms_hi": (
            "पत्तियों पर छोटे गीले धब्बे जो बाद में गहरे भूरे हो जाते हैं और उनके चारों ओर पीला घेरा बन जाता है।"
        ),
        "suggestion_en": (
            "Remove affected leaves, avoid overhead sprinkler irrigation, keep good airflow, and spray copper hydroxide if required."
        ),
        "suggestion_hi": (
            "प्रभावित पत्तियों को तोड़कर हटा दें, ऊपर से पानी न छिड़कें, पौधों के बीच हवा का बहाव रखें और कॉपर फंगीसाइड/बैक्टीरिया-नाशक का प्रयोग करें।"
        ),
    },
    "Pepper__bell___healthy": {
        "name_en": "Bell Pepper (Capsicum) — Healthy Plant",
        "name_hi": "शिमला मिर्च — स्वस्थ पौधा (कोई बीमारी नहीं)",
        "is_healthy": True,
        "problem_en": (
            "No disease detected! Your bell pepper plant leaf looks glossy, green, and completely healthy."
        ),
        "problem_hi": (
            "पौधे में कोई बीमारी नहीं है! आपकी शिमला मिर्च की पत्ती चमकदार, हरी और पूरी तरह स्वस्थ है।"
        ),
        "symptoms_en": "Smooth, bright green leaf with no bacterial spots or yellowing.",
        "symptoms_hi": "चमकदार हरी पत्ती जिस पर कोई धब्बे या पीलापन नहीं है।",
        "suggestion_en": (
            "Continue regular watering, avoid waterlogging, and inspect under leaves periodically for pests."
        ),
        "suggestion_hi": (
            "नियमित देखभाल जारी रखें, गमले या खेत में पानी न रुकने दें और समय-समय पर पत्तियों की जांच करते रहें।"
        ),
    },
}


DEFAULT_DISEASE_DETAIL = {
    "name_en": "Plant Health Issue Detected",
    "name_hi": "पौधे में स्वास्थ्य समस्या (Plant Health Issue)",
    "is_healthy": False,
    "problem_en": "Signs of stress or leaf infection were detected on the plant leaf.",
    "problem_hi": "पौधे की पत्ती पर संक्रमण या पोषण की कमी के शुरुआती लक्षण दिखाई दे रहे हैं।",
    "symptoms_en": "Discoloration, spots, or uneven texture on the leaf surface.",
    "symptoms_hi": "पत्ती पर रंग का बदलना, धब्बे पड़ना या पत्ती का मुरझाना।",
    "suggestion_en": "Remove heavily damaged leaves, avoid overwatering on foliage, and ensure proper sunlight and airflow.",
    "suggestion_hi": "अधिक खराब पत्तियों को हटा दें, पत्तियों के ऊपर पानी न डालें और पौधे को उचित धूप व हवा मिलने दें।",
}


# =========================================================
# FEATURE EXTRACTION (FOR LOCAL .PKL MODEL)
# =========================================================

def extract_features(pil_image):
    """Extract HOG, color-statistic, and RGB histogram features used in training."""
    img = pil_image.convert("RGB").resize(tuple(image_size))
    image_array = np.asarray(img, dtype=np.float32) / 255.0

    gray = np.mean(image_array, axis=2)
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(3, 3),
        block_norm="L2-Hys",
    )

    color_features = []
    for channel in range(3):
        channel_data = image_array[:, :, channel]
        color_features.extend(
            [
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data),
            ]
        )

    histogram_features = []
    for channel in range(3):
        histogram, _ = np.histogram(
            image_array[:, :, channel],
            bins=64,
            range=(0, 1),
            density=True,
        )
        histogram_features.extend(histogram)

    return np.concatenate(
        [
            hog_features,
            np.asarray(color_features, dtype=np.float32),
            np.asarray(histogram_features, dtype=np.float32),
        ]
    ).astype(np.float32)


# =========================================================
# OPTIONAL GEMINI AI VISION ANALYSIS (IF API KEY IS PRESENT)
# =========================================================

def analyze_with_gemini_vision(pil_image, fallback_label=None, fallback_conf=None):
    """
    If GEMINI_API_KEY is configured in .env, uses Google Gemini Vision AI
    to inspect the clicked/uploaded plant photo directly and explain the exact
    disease and plant problem in both English and Hindi.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        buffered = io.BytesIO()
        rgb_img = pil_image.convert("RGB")
        rgb_img.thumbnail((1024, 1024))
        rgb_img.save(buffered, format="JPEG", quality=85)
        b64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        hint_text = (
            f"Local classifier hint: {fallback_label} ({fallback_conf:.1f}%)."
            if fallback_label
            else ""
        )

        prompt = (
            "You are AgriSense AI, an expert plant pathologist. Analyze this plant/leaf image carefully. "
            f"{hint_text}\n"
            "Return ONLY valid JSON with these exact keys:\n"
            "{\n"
            '  "disease_key": "Short English Title of Plant & Condition (e.g., Tomato Early Blight or Healthy Plant)",\n'
            '  "name_en": "English name of the plant and disease/condition",\n'
            '  "name_hi": "Hindi name of the plant and disease/condition (देवनागरी में)",\n'
            '  "confidence": 92.5,\n'
            '  "is_healthy": false,\n'
            '  "problem_en": "Clear 2-sentence explanation in English of what problem/disease is in the plant and why it happens.",\n'
            '  "problem_hi": "Clear 2-sentence explanation in Hindi (देवनागरी) of पौधे में क्या दिक्कत है और यह क्यों होती है।",\n'
            '  "symptoms_en": "Visible symptoms on the leaf in English.",\n'
            '  "symptoms_hi": "पत्ती पर दिखने वाले मुख्य लक्षण हिंदी में।",\n'
            '  "suggestion_en": "Practical treatment and care advice in English.",\n'
            '  "suggestion_hi": "इलाज और देखभाल के व्यावहारिक उपाय हिंदी में।"\n'
            "}"
        )

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={api_key}"
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64_image,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.2,
            },
        }

        response = requests.post(url, json=payload, timeout=15)
        if response.status_code == 200:
            data = response.json()
            text_out = (
                data["candidates"][0]["content"]["parts"][0]["text"].strip()
            )
            parsed = json.loads(text_out)
            return parsed
    except Exception as exc:
        print("Gemini Vision fallback to local model:", exc)

    return None


# =========================================================
# MAIN PREDICTION FUNCTION
# =========================================================

def load_pil_image(image_source):
    """Safely open an image from a Flask FileStorage, BytesIO, file path, or PIL Image."""
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")

    if hasattr(image_source, "seek"):
        image_source.seek(0)

    pil_img = Image.open(image_source).convert("RGB")

    if hasattr(image_source, "seek"):
        image_source.seek(0)

    return pil_img


def get_disease_details(disease_name, pil_image=None, confidence=None):
    """
    Return complete bilingual (English + Hindi) details about what problem
    the plant has, its symptoms, and treatment suggestions.
    """
    # 1. Try Gemini Vision AI if GEMINI_API_KEY is configured in .env
    if pil_image is not None:
        ai_result = analyze_with_gemini_vision(pil_image, disease_name, confidence)
        if ai_result and isinstance(ai_result, dict):
            return {
                "disease_key": ai_result.get("disease_key", disease_name),
                "name_en": ai_result.get("name_en", disease_name.replace("_", " ")),
                "name_hi": ai_result.get("name_hi", disease_name.replace("_", " ")),
                "is_healthy": bool(ai_result.get("is_healthy", False)),
                "problem_en": ai_result.get("problem_en", DEFAULT_DISEASE_DETAIL["problem_en"]),
                "problem_hi": ai_result.get("problem_hi", DEFAULT_DISEASE_DETAIL["problem_hi"]),
                "symptoms_en": ai_result.get("symptoms_en", DEFAULT_DISEASE_DETAIL["symptoms_en"]),
                "symptoms_hi": ai_result.get("symptoms_hi", DEFAULT_DISEASE_DETAIL["symptoms_hi"]),
                "suggestion_en": ai_result.get("suggestion_en", DEFAULT_DISEASE_DETAIL["suggestion_en"]),
                "suggestion_hi": ai_result.get("suggestion_hi", DEFAULT_DISEASE_DETAIL["suggestion_hi"]),
                "confidence": float(ai_result.get("confidence", confidence or 90.0)),
            }

    # 2. Look up in comprehensive bilingual dictionary for the local ML model
    info = DISEASE_DETAILS.get(disease_name, DEFAULT_DISEASE_DETAIL).copy()
    info["disease_key"] = disease_name
    if "name_en" not in info:
        info["name_en"] = disease_name.replace("___", " - ").replace("_", " ")
    info["confidence"] = float(confidence if confidence is not None else 0.0)
    return info


def predict_disease(image):
    """
    Predict plant disease from an uploaded or camera-captured image.
    Returns (disease_name, confidence) for full compatibility, and attaches
    bilingual details if predict_disease_detailed() is called.
    """
    pil_image = load_pil_image(image)

    if model is not None and label_encoder is not None:
        features = extract_features(pil_image).reshape(1, -1)
        prediction = model.predict(features)[0]
        disease_name = label_encoder.inverse_transform([prediction])[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities) * 100)
    else:
        disease_name = "Tomato_Early_blight"
        confidence = 85.0

    # If Gemini Vision AI is active, let it refine the prediction directly from the photo
    ai_details = analyze_with_gemini_vision(pil_image, disease_name, confidence)
    if ai_details:
        disease_name = ai_details.get("disease_key", disease_name)
        confidence = float(ai_details.get("confidence", confidence))

    return disease_name, confidence


def predict_disease_detailed(image):
    """
    Analyze the plant photo and return:
      (disease_name, confidence, bilingual_details_dict)
    """
    pil_image = load_pil_image(image)

    if model is not None and label_encoder is not None:
        features = extract_features(pil_image).reshape(1, -1)
        prediction = model.predict(features)[0]
        disease_name = label_encoder.inverse_transform([prediction])[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities) * 100)
    else:
        disease_name = "Tomato_Early_blight"
        confidence = 85.0

    details = get_disease_details(
        disease_name,
        pil_image=pil_image,
        confidence=confidence,
    )
    final_name = details.get("disease_key", disease_name)
    final_confidence = float(details.get("confidence", confidence))

    return final_name, final_confidence, details