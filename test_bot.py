import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page ki settings (Title aur Icon)
st.set_page_config(page_title="Gold & Jewelry Shop AI", page_icon="✨", layout="centered")

# Website ka Main Header
st.title("✨ Gold & Jewelry Shop")
st.subheader("Ahmad Raza Sahib's AI Assistant")
st.write("Sarafa Bazar mein khush aamdeed! Aap Sone ke rate, timing, ya shop ke baare mein kuch bhi pooch sakte hain.")
st.markdown("---")

# 1. Mukammal Dataset
data = {
    'Question': [
        "Aap ki shop ka naam kya hai?", "Dukaan ka naam batayein?", "Owner kaun hai?", "Ahmad Raza ki shop ka kya naam hai?",
        "Shop ka address kya hai?", "Dukaan kahan par hai?", "Sarafa bazar mein aap ki shop kahan hai?", "Aap se rabta kaise karein?",
        "Contact number kya hai?", "Mobile number bata dein?", "Dukaan kab khulti hai?", "Timing kya hai dukaan ki?",
        "Kya friday ko shop open hoti hai?", "Friday ko open hai?", "Aap kaunse karat ka sona bechte hain?", "Kya 22 karat gold mil jaye ga?",
        "Mazdoori kitni hoti hai?", "Making charges kya hain?", "Kya aap order par jewelry banate hain?", "Custom design banwa sakte hain?",
        "Gold rate check karne ka kya tareeqa hai?", "Aaj ka sone ka rate kya hai?", "10 gram gold kitne ka hai?", "Tola ka rate kya chal raha hai?",
        "Kya aap log 24 karat gold bechte hain?", "24K gold mil jayega aap ki shop se?", "Ahmad Raza sahib ki shop par kaunse designs hain?",
        "Shaadi ke liye jewelry set banwana hai?", "Rings ke designs dekhne hain?", "Kya aap log chandi (silver) bhi bechte hain?",
        "Sone ki purity kaise check karte hain?", "Kya aap ke gold ki guarantee hoti hai?", "Purana sona bechne ka kya tareeqa hai?",
        "Gold exchange policy kya hai aap ki?", "Kya purana sona wapas lete hain?", "Jewelry ki mazdoori fix hoti hai?",
        "Karat kam hone se mazdoori badalti hai?", "Kya credit card se payment ho sakti hai?", "Online bank transfer accept karte hain?",
        "Dukaan par aane ke liye appointment chahiye?", "Kya Ahmad Raza sahib shop par maujood hote hain?", "Shop ka poora naam dubara bata dein?",
        "Aap ki shop kaal se chal rahi hai?", "Customized design mein kitna waqt lagta hai?", "Kya aap log delivery bhi karte hain?",
        "Ahmad raza sb ki shop ka off kab hota hai?", "Kya juma ko dukaan khuli hoti hai?", "Juma ke din timing kya hai?",
        "Ahmad Raza ki shop juma ko open hoti hai?", "Aap ki shop Sarafa Bazar mein kis jagah waqe hai?", "Sarafa Bazar wali dukaan ka rasta bata dein?",
        "Address kya hai aap ki gold shop ka?", "Gold and Jewelry shop ka pata bata dein?", "Kya dukaan ka koi phone number hai?",
        "Ahmad Raza sahib ka contact number kya hai?", "WhatsApp par rabta karne ka number?", "Dukaan ka mobile number chahiye tha?",
        "Aap ki shop par kaun kaun se karat ka sona milta hai?", "Kya aap ke paas har karat ka gold hota hai?", "Kaunse karat ka sona bechte hain Ahmad Raza?",
        "Sona kis karat ka mil jaye ga?", "Jewelry banane ki mazdoori kya chal rahi hai?", "Making charges har cheez par same hoti hain?",
        "Mazdoori kis tarah hisab hoti hai?", "Gold design ke making charges kya hain?", "Kya aap apni marzi ka design banwa sakte hain?",
        "Order par sona banwane ka kya tareeqa hai?", "Customized gold jewelry ban jati hai?", "Kya aap log custom orders lete hain?",
        "Aaj ka sone ka rate kya chal raha hai?", "Gold ka live rate bata dein?", "Aaj per tola gold price kya hai?",
        "10 gram sone ki kya qeemat hai aaj?", "Kya aap ke paas shaadi ke sets hote hain?", "Bridal jewelry mil jaye gi?",
        "Wedding jewelry ke designs hain?", "Shaadi ka set banwana ho toh kitna time lagta hai?", "Kya aap log sonay ki churiyan bechte hain?",
        "Gold bangles ke designs dikha dein?", "Kangan banwane hain gold ke?", "Bangles order par ban jati hain?",
        "Engagement ke liye ring mil jaye gi?", "Sone ki ring dekhni hai?", "Gents gold rings banti hain aap ki shop par?",
        "Ladie rings ke designs hain?", "Kya aap gold chains bhi bechte hain?", "Sone ka locket set mil jayega?",
        "Gold chain ke designs dekhne hain?", "Light weight jewelry mil sakti hai?", "Roz pehn'ne wali jewelry mil jaye gi?",
        "Kya aap log chandi ka kaam karte hain?", "Silver jewelry mil sakti hai?", "Chandi ke bartan banwane hon toh?",
        "Sone ki purity ki kya guarantee hoti hai?", "Kya aap log guarantee card dete hain?", "Sona asli hai ya nakli kaise pata chalega?",
        "Computerized testing hoti hai aap ki dukaan par?", "Kya purana sona becha ja sakta hai aap ki shop par?", "Purani jewelry exchange ho sakti hai?",
        "Sona wapas karne par kitni kaat hoti hai?", "Gold exchange policy kya hai Ahmad Raza ki shop ki?", "Kya payment credit card se ki ja sakti hai?",
        "Online payment accept karte hain aap log?", "Kya main check ke zariye payment kar sakta hoon?", "Card swipe ki sahulat hai?",
        "Dukaan par aane ke liye koi appointment leni parti hai?", "Kya pehle se time fix karna zaroori hai dukaan aane ke liye?",
        "Ahmad Raza sahib kis waqt shop par hote hain?", "Kya meri baat direct Ahmad Raza sahib se ho sakti hai?", "Gold and Jewelry shop kab se bani hui hai?",
        "Aap ki shop kitni purani hai?", "Kya aap log home delivery karte hain?", "Gold ki delivery ghar par mil sakti hai?",
        "Order dene ke liye kitne paise pehle dene hote hain?", "Biyana (advance) dena lazmi hai order par?", "Order cancel karne par advanced wapas milega?",
        "Heavy jewelry banne mein kitne din lagte hain?", "Simple ring kitne din mein ban jati hai?", "Kya aap investment ke liye gold biscuits bechte hain?",
        "Sona biscuit form mein mil jayega?", "Investment ke liye konsa sona sab se acha hai?", "Dukaan par sab se zyada rash kis waqt hota hai?",
        "Sukoon se design dekhne ke liye kis waqt aayein?", "Aap ki shop ki sab se khas baat kya hai?", "Ahmad Raza ki shop par log kyun aate hain?",
        "Sone ka rate barhne wala hai ya kam hone wala hai?", "Future mein gold rate kya hoga?", "Kya aap ke paas jhumkon ke designs hain?",
        "Gold tops mil jayenge?", "Kya aap log bacchon ki jewelry bhi banate hain?", "Bacchon ke liye gold items hain?",
        "Kya gold rate pure Pakistan mein same hota hai?", "Karachi aur Lahore ka gold rate same hota hai?", "Gold par tax kitna hota hai?",
        "Bill par tax lagta hai gold ke?", "Kya aap bina bill ke sona bechte hain?", "Raseed gum ho jaye toh kya hoga?",
        "Purani raseed dikha kar sona exchange ho jayega?", "Kya aap log diamonds mein bhi deal karte hain?", "Diamond ring mil sakti hai order par?",
        "Gold par polish dubara ho sakti hai?", "Jewelry saaf karwani ho toh charges kya hain?", "Kala sona kya hota hai?",
        "Kya aap log pure white gold bechte hain?", "White gold aur yellow gold mein kya farq hai?", "Aap ki shop par safety ke kya intezamat hain?",
        "CCTV cameras lagay hue hain shop par?", "Kya Ahmad Raza sahib ke paas karigar dukaan par hi hote hain?", "Karigar apna kaam khud karte hain ya bahar se hota hai?",
        "Dukaan par aane ka address rasta dubara samjha dein?", "Ahmad Raza jewellers kahan hai?", "Online catalog dekhne ka koi tareeqa hai?",
        "WhatsApp par designs ki photos mil sakti hain?", "Kya aap log kangan par nagine (stones) lagate hain?", "Real stones wali jewelry mil sakti hai?",
        "Nagine lagane ke charges alag hote hain?", "Gold rate subah kis time aata hai?", "Market rate kis time khulta hai rozana?",
        "Aap ki shop se sona khareedne ka kya faida hai?", "Trustworthy shop kyun hai yeh?", "Kya aap log wholesale mein bhi deal karte hain?",
        "Retail shop hai ya wholesale?", "Sone ka rate upar kyun jata hai?", "Sona sasta kab hoga?",
        "Kya aap log nose pin (laung) bechte hain?", "Nose pin gold ki mil jaye gi?", "Bracelets ke designs hain aap ke paas?",
        "Gold bracelet banwana ho toh design mil jayenge?", "Kya aap log sona girwi (mortgage) rakhte hain?", "Sona rakh kar paise udhaar mil sakte hain?",
        "Kya installment par jewelry milti hai?", "Qiston par sona mil sakta hai Ahmad Raza ki shop se?", "Aap ki shop par kaunse designs sab se zyada bikte hain?",
        "Latest jewelry design kaunse aaye hain?", "Sone ka rate check karne ka website link hai koi?", "Aap log roz ka rate kahan se dekhte hain?",
        "Kya 22 karat gold asli hota hai?", "24K aur 22K mein kya farq hai?", "21 karat gold mein kitna sona hota hai?",
        "18 karat gold sasta kyun hota hai?", "Kya 18K gold jaldi kharab ho jata hai?", "Sone ka set saaf karne ka tarika kya hai ghar par?",
        "Jewelry dho dho kar kharab toh nahi hoti?", "Kya aap log raseed par karat likhte hain?", "Raseed par weight sahi likha hota hai?",
        "Weight check karne ka kanda accurate hai?", "Ahmad Raza ki shop Sarafa bazar mein kis jagah hai?", "Gold and Jewelry shop ka address kya hai?",
        "Kya mujhe aap ki shop ka location mil sakta hai?", "Aap ki dukaan kis sheher mein hai?", "Ahmad Raza sahib se rabta karne ka number?",
        "Kya is number par WhatsApp maujood hai?", "Shop ka contact number kya hai?", "Emergency mein kis number par call karein?",
        "Aap se baat karne ka waqt kya hai?", "Friday ko shop kyun band hoti hai?", "Kya sunday ko shop khuli hoti hai?",
        "Sunday ki timing kya hai?", "Weekend par shop open hoti hai?", "Monday ko kis time shop khulti hai?",
        "Thursday ko shop khuli hogi?", "Kya kal shop open hai?", "Gold rate din mein kitni baar badalta hai?",
        "Kya aap mujhe abhi ka gold rate bata sakte hain?", "Kya aap log 21 karat gold bechte hain?", "18 karat gold mil jayega?",
        "Karat mein kya farq hota hai?", "Khalis sona kaunsa hota hai?", "Jewelry ke liye sab se behtareen karat kaunsa hai?",
        "Bangles ke designs hain aap ke paas?", "Necklace set banwana ho toh?", "Locket ke designs dekhne hain?",
        "Earrings mil jayenge?", "Kya aap log gents rings banate hain?", "Order par jewelry kitne din mein tayyar hoti hai?",
        "Kya main apni pasand ka design laa sakte hain?", "Agar order emergency mein chahiye ho toh?", "Order dene ke liye advanced payment karni hoti hai?",
        "Kya aap sone ki purity ka certificate dete hain?", "Agar sone mein milaawat ho toh?", "Kya aap purani jewelry ka gold check kar sakte hain?",
        "Karat test karne ke kya charges hain?", "Sone par kaat kitni hoti hai?", "Purana sona exchange karne par naye design ki mazdoori deni hogi?",
        "Kya aap log chandi ke bartan ya jewelry lete hain?", "Payment ke liye cash zaroori hai?", "Kya check accept karte hain?",
        "Dukaan par rash kab hota hai?", "Kya Ahmad Raza sahib se direct baat ho sakti hai?", "Gold and Jewelry Shop kab se bani hui hai?",
        "Aap ki shop ki kya khususiyaat hain?", "Kya aap ke paas choti jewelry bhi hoti hai?", "Karat gold biscuit mil jayenge?",
        "Sona khareedna investment ke liye kaisa hai?"
    ],
    'Answer': [
        "Hamari shop ka naam \"Gold and Jewelry Shop\" hai aur iske malik Ahmad Raza sahib hain.", "Hamari shop ka naam \"Gold and Jewelry Shop\" hai aur iske malik Ahmad Raza sahib hain.", "Is shop ke malik Ahmad Raza sahib hain.", "Ahmad Raza sahib ki shop ka naam \"Gold and Jewelry Shop\" hai.",
        "Hamari shop Sarafa Bazar mein waqe hai.", "Hamari shop Sarafa Bazar mein waqe hai.", "Aap Sarafa Bazar aakar hamari shop \"Gold and Jewelry Shop\" par tashreef laa sakte hain.", "Aap hamare phone number 0313******** par call ya WhatsApp kar sakte hain.",
        "Aap hamare phone number 0313******** par call ya WhatsApp kar sakte hain.", "Aap hamare phone number 0313******** par call ya WhatsApp kar sakte hain.", "Shop Monday se Thursday aur Saturday se Sunday tak khuli hoti hai. Friday ko chutti hoti hai.", "Shop Monday se Thursday aur Saturday se Sunday tak khuli hoti hai. Friday ko chutti hoti hai.",
        "Nahi, Friday ko hamari shop off (band) hoti hai.", "Nahi, Friday ko hamari shop off (band) hoti hai.", "Hum har karat (24K 22K 21K aur 18K) ka sona bechte hain.", "Ji haan, hamare paas har karat (including 22K, 24K, 18K) ka gold dastiyab hai.",
        "Sone ki making charges (mazdoori) jewelry ke design ke hisab se badalti hain.", "Sone ki making charges (mazdoori) jewelry ke design ke hisab se badalti hain.", "Ji bilkul! Hum aap ki pasand ke mutabiq order par bhi jewelry tayyar karte hain.", "Ji bilkul! Hum aap ki pasand ke mutabiq order par bhi jewelry tayyar karte hain.",
        "Hamari shop par rozana market ke mutabiq sonay ka taza tareen rate bataya jata hai. Aap call kar ke bhi pooch sakte hain.", "Sone ka rate rozana badalta hai. Taza tareen rate jaan'ne ke liye aap hamare number 0313******** par raabta kar sakte hain.", "Gold ka rate market ke hisab se up-down hota rehta hai. Is waqt ka live rate aap call par jaan sakte hain.", "Tola ka rate market ke mutabiq rozana subah set hota hai. Sahi maloomat ke liye dukaan par rabta karein.",
        "Ji haan! Hamare paas 24 karat ka khalis sona har waqt dastiyab hota hai.", "Ji bilkul, Gold and Jewelry Shop par aap ko 24 karat gold biscuits aur jewelry mil jayegi.", "Ahmad Raza sahib ki shop par shaadi ke sets, rings, bangles, aur jhumkon ke latest designs maujood hain.",
        "Hum shaadi (bridal) ke mukammal jewelry sets order par bohot khubsoorat tayyar karte hain.", "Aap hamari shop Sarafa Bazar tashreef laayein, hum aap ko rings ke saare designs dikha denge.", "Hum mukammal taur par gold aur gold jewelry ka kaam karte hain, lekin special order par chandi ka kaam bhi ho jata hai.",
        "Hum customer ke samne computerized testing machine se sone ki purity check kar ke tasalli karwate hain.", "Ji haan, hamari shop se liye gaye har sone par pure hone ki mukammal official raseed aur guarantee di jaati hai.", "Agar aap purana sona bech kar naya lena chahein, toh hum uski purity check kar ke market rate ke mutabiq behtareen qeemat dete hain.",
        "Aap apna purana sona laa kar hamari shop se naye designs ke sath exchange kar sakte hain, kaat market ke rules ke mutabiq hoti hai.", "Ji haan, hum purana sona wapas bhi lete hain aur exchange bhi karte hain.", "Nahi, har jewelry ka design alag hota hai is liye mazdoori (making charges) design ki bariki ke hisab se tay hoti hai.",
        "Mazdoori ka taluq karat se nahi balkay design ki mehnat aur bariki se hota hai.", "Ji bilkul, hamari shop par cash ke sath sath credit card aur online bank transfer ki sahulat bhi maujood hai.", "Ji haan, aap online banking ya bank transfer ke zariye bhi payment kar sakte hain.",
        "Aap bina appointment ke bhi Monday se Sunday (Friday ke ilawa) kisi bhi waqt Sarafa Bazar aa sakte hain.", "Ji haan, Ahmad Raza sahib aksar auqat shop par hi maujood hote hain aur customers ko khud deal karte hain.", "Shop ka poora naam \"Gold and Jewelry Shop\" hai.",
        "Hamari shop Sarafa Bazar mein ek purani aur mutamad (trusted) shop hai jo saalon se khalis sone ka kaam kar rahi hai.", "Order par jewelry banwane mein design ke hisab se 1 se 2 hafte ka waqt lag sakta hai.", "Sone ki security ki wajah se hum sirf shop par hi dealing karte hain, home delivery ki sahulat nahi hai.",
        "Shop Monday se Thursday aur Saturday se Sunday tak khuli hoti hai. Friday ko chutti hoti hai.", "Nahi, Friday ko hamari shop off (band) hoti hai.", "Friday ko hamari shop mukammal band hoti hai, baki din khuli hoti hai.",
        "Nahi, Ahmad Raza sahib ki shop Friday (Juma) ko off hoti hai.", "Hamari shop Sarafa Bazar mein waqe hai. Aap market aakar \"Gold and Jewelry Shop\" ka pooch sakte hain.", "Aap Sarafa Bazar aakar hamari shop \"Gold and Jewelry Shop\" par tashreef laa sakte hain ya 0313******** par call kar lein.",
        "Hamari shop Sarafa Bazar mein waqe hai.", "Hamari shop Sarafa Bazar mein waqe hai.", "Ji haan, aap hamare phone number 0313******** par call ya WhatsApp kar sakte hain.",
        "Ahmad Raza sahib ka number 0313******** hai, aap call ya WhatsApp kar sakte hain.", "Aap hamare phone number 0313******** par WhatsApp message kar sakte hain.", "Hamari shop ka number 0313******** hai.",
        "Hum har karat (24K 22K 21K aur 18K) ka sona bechte hain.", "Ji haan, hamare paas har karat (24K, 22K, 21K, aur 18K) ka gold dastiyab hai.", "Ahmad Raza sahib ki shop par har karat (24K, 22K, 21K, aur 18K) ka sona bechha jata hai.",
        "Hum har karat (24K 22K 21K aur 18K) ka sona bechte hain.", "Sone ki making charges (mazdoori) jewelry ke design ke hisab se badalti hain.", "Nahi, making charges (mazdoori) jewelry ke design ki bariki aur mehnat ke hisab se badalti hain.",
        "Sone ki making charges (mazdoori) jewelry ke design ke hisab se badalti hain.", "Sone ki making charges (mazdoori) jewelry ke design ke hisab se badalti hain.", "Ji bilkul! Hum aap ki pasand ke mutabiq order par bhi jewelry tayyar karte hain.",
        "Aap apna pasandida design laa kar order de sakte hain, hum bilkul waisa hi tayyar kar dein ge.", "Ji bilkul! Hum aap ki pasand ke mutabiq order par bhi jewelry tayyar karte hain.", "Ji bilkul! Hum aap ki pasand ke mutabiq order par bhi jewelry tayyar karte hain.",
        "Sone ka rate rozana badalta hai. Taza tareen rate jaan'ne ke liye aap hamare number 0313******** par raabta kar sakte hain.", "Live gold rate har lamha badalta hai isliye naye rate ke liye aap hamare number 0313******** par foran call karein.", "Gold ka rate market ke hisab se badalta rehta hai. Is waqt ka live rate aap call par jaan sakte hain.",
        "Sone ka rate rozana badalta hai. Taza tareen rate jaan'ne ke liye aap hamare number 0313******** par raabta kar sakte hain.", "Ji haan! Gold and Jewelry Shop par shaadi ke mukammal bridal sets order par bohot khubsoorat tayyar kiye jaate hain.", "Ji haan! Hum shaadi (bridal) ke mukammal jewelry sets order par bohot khubsoorat tayyar karte hain.",
        "Ji haan! Hamari shop par shaadi ke sets, rings, bangles, aur jhumkon ke latest designs maujood hain.", "Heavy bridal set ho toh design ke hisab se 10 se 15 din lag sakte hain.", "Ji haan! Gold and Jewelry Shop par bangles (churiyan) aur kangan ke behtareen designs dastiyab hain.",
        "Aap hamari shop Sarafa Bazar tashreef laayein, hum aap ko bangles aur baqi saare designs dikha denge.", "Ji haan! Hamare paas kangan aur churiyon ke behtareen designs maujood hain aur order par bhi bante hain.", "Ji haan! Hum aap ki pasand aur size ke mutabiq bangles order par tayyar karte hain.",
        "Ji haan! Hamari shop par engagement aur nikah ke liye khubsoorat gold rings ke designs maujood hain.", "Aap hamari shop Sarafa Bazar tashreef laayein, hum aap ko rings ke saare designs dikha denge.", "Ji haan! Hum mardon ke liye gold ya silver mein customized rings bhi tayyar karte hain.",
        "Ji haan! Hamari shop par ladies rings aur engagement rings ke bohot se latest designs dastiyab hain.", "Ji haan! Hamari shop par har weight mein khoobsurat chains aur lockets ke designs maujood hain.", "Ji haan! Hamere paas light weight aur heavy locket sets ke behtareen designs dastiyab hain.",
        "Aap shop par aakar chains aur lockets ke latest designs dekh sakte hain.", "Ji haan! Hamare paas light weight rings, chains aur earrings bhi hote hain jo rozana pehn'ne ke liye behtareen hain.", "Ji haan! Hamare paas light weight aur daily-wear jewelry ke bohot pyare designs hain.",
        "Hum mukammal taur par gold ka kaam karte hain, lekin special order par chandi (silver) ka kaam bhi ho jata hai.", "Chandi ka kaam hamari shop par sirf special orders par kiya jata hai, aam taur par hum gold mein deal karte hain.", "Aap shop par aakar rabta kar sakte hain, special order par chandi ka kaam ho jata hai.",
        "Ji haan, hamari shop se liye gaye har sone par pure hone ki mukammal official raseed aur guarantee di jaati hai.", "Ji bilkul! Hamari shop se khareede gaye har gold item ki official raseed aur guarantee certificate diya jata hai.", "Hum customer ke samne computerized testing machine se sone ki purity check kar ke tasalli karwate hain.",
        "Ji haan! Hum customer ke samne computerized testing machine se sone ki purity check kar ke tasalli karwate hain.", "Ji haan, hum purana sona wapas bhi lete hain aur market rate ke mutabiq behtareen qeemat dete hain.", "Aap apna purana sona laa kar hamari shop se naye designs ke sath exchange kar sakte hain, kaat market ke rules ke mutabiq hoti hai.",
        "Sona bechte ya exchange karte waqt kaat market ke usoolon aur sone ki halat ko dekh kar tay ki jaati hai.", "Aap apna purana sona laa kar exchange kar sakte hain. Kaat market ke mutabiq hogi aur naye design ki mazdoori alag hogi.", "Ji bilkul, hamari shop par cash ke sath sath credit card aur online bank transfer ki sahulat bhi maujood hai.",
        "Ji haan, aap online banking ya bank transfer ke zariye bhi payment kar sakte hain.", "Check clear hone ke baad hi jewelry deliver ki jaati hai isliye cash ya online transfer ko tarjeeh di jaati hai.", "Ji haan, hamari shop par credit aur debit card swipe karne ki sahulat maujood hai.",
        "Aap bina appointment ke bhi Monday se Sunday (Friday ke ilawa) kisi bhi waqt Sarafa Bazar aa sakte hain.", "Bina appointment ke aap kisi bhi waqt aa sakte hain, hamari shop open hoti hai.",
        "Ahmad Raza sahib aksar auqat shop par hi maujood hote hain aur customers ko khud deal karte hain.", "Ji bilkul! Aap dukaan par aakar direct Ahmad Raza sahib se mil sakte hain ya unke number par baat kar sakte hain.", "Yeh Sarafa Bazar ki aik jani mani aur purani shop hai jo ke aala quality aur aitmad ka doosra naam hai.",
        "Yeh Sarafa Bazar ki aik jani mani aur purani shop hai jo ke aala quality aur aitmad ka doosra naam hai.", "Sone ki security ki wajah se hum sirf shop par hi dealing karte hain, home delivery ki sahulat nahi hai.", "Sone ki security ki wajah se hum home delivery nahi karte, aap ko shop par hi aana hoga.",
        "Order confirm karne ke liye kuch advanced (biyana) dena hota hai aur baqi payment delivery ke waqt hoti hai.", "Ji haan! Order confirm karne ke liye kuch advanced (biyana) dena hota hai.", "Order cancel karne ki surat mein karigar ki mehnat ki katoti kar ke baqi faisla market policy ke mutabiq hota hai.",
        "Heavy bridal ya wedding sets banne mein zyadatar 10 se 15 din ka waqt lag sakta hai.", "Simple design jaise ring ya tops 5 se 7 din mein tayyar ho jaate hain.", "Ji haan! Investment ke liye 24K gold biscuits aur baars bhi hamari shop se mil jaate hain.",
        "Ji haan! Hamare paas investment ke liye 24K pure gold biscuits maujood hote hain.", "Investment ke liye 24 karat (24K) gold biscuit ya baars sab se behtareen hote hain.", "Shaam ke waqt dukaan par zyadatar rash hota hai agar aap tasalli se design dekhna chahte hain toh doper ke waqt tashreef laayein.",
        "Aap Monday se Thursday doper 2 baje se shaam 6 baje ke darmayan aayein toh sukoon se dealing ho sakti hai.", "Hamari khususiyaat mein khalis sona, jadeed designs, behtareen karigari aur munasib mazdoori shamil hain.", "Ahmad Raza sahib ki shop apni imandari, pure gold aur aala quality ke designs ki wajah se jani jati hai.",
        "Gold rate international market aur dollar ki qeemat ke mutabiq badalta rehta hai is bare mein pehle se kuch kehna mushkil hota hai.", "Gold rate ke baare mein pehle se andaza lagana mushkil hai kyunki yeh rozana market ke hisab se up-down hota hai.", "Ji haan! Hamare paas her tarah ke earrings, jhumkay aur tops ke designs maujood hain.",
        "Ji haan! Hamari shop par rozana pehn'ne ke liye light weight gold tops maujood hain.", "Ji haan! Hum bacchon ke liye choti rings, bangles aur doreen (chains) bhi tayyar karte hain.", "Ji haan, bacchon ke liye light weight gold rings aur bracelets dastiyab hain.",
        "Zyadatar shehron mein thoda bohot farq hota hai kyunki har sheher ki sarafa association apna rate thoda alag set kar sakti hai.", "Baaz auqat shehron ke rates mein kuch saikron (hundreds) ka farq hota hai, aam taur par thoda bohat diffence hota hai.", "Sone par tax hukoomat ki mojudah policy aur raseed ke mutabiq lagaya jata hai jo ke bill mein shamil hota hai.",
        "Ji haan, hukoomat ke qanoon ke mutabiq gold jewelry ke bill par muqarrar tax lagaya jata hai.", "Nahi, hamari shop par har deal proper raseed aur bill ke sath hoti hai taake customer ka record mehfooz rahe.", "Agar raseed gum ho jaye toh hum apne computer record se check kar ke verify kar sakte hain lekin raseed sambhal kar rakhna behtar hai.",
        "Ji haan! Agar hamari shop ki purani raseed maujood ho toh exchange process mazeed aasan aur fast ho jata hai.", "Hum zyadatar gold aur gold jewelry mein deal karte hain, diamond jewelry sirf special order par hi banayi jati hai.", "Ji haan, agar aap order dein toh hum aala quality ke diamonds lagakar ring tayyar karwa sakte hain.",
        "Ji haan! Agar aap ki jewelry purani lag rahi hai toh hum us par dubara polish kar ke bilkul naya jaisa kar dete hain.", "Hamare purane customers ke liye jewelry ki safai aur polish bilkul munasib charges ya free mein ki jaati hai.", "Sona kabhi kala nahi hota, baaz auqat sweat (paseene) ya chemicals ki wajah se polish dundli ho jati hai jo wash karne se theek ho jati hai.",
        "Ji haan, special order par 22K ya 18K white gold ki jewelry bhi tayyar ki jaati hai.", "Donon mein sone ki miqdar barabar ho sakti hai, bas white gold mein silver ya palladium mix kar ke uska rang badla jata hai.", "Hamari shop Sarafa Bazar mein hai aur yahan security ke behtareen intezamat aur CCTV cameras lagaye gaye hain.",
        "Ji haan, customer aur shop ki safety ke liye mukammal CCTV monitoring hoti hai.", "Ji haan, hamare expert karigar workshop mein hotey hain jo shop ke sath hi judi hui hai.", "Hamare paas apne expert aur khususi karigar maujood hain jo har design ko bariki se khud tayyar karte hain.",
        "Shop Sarafa Bazar ke main chowk ke paas waqe hai. Aap kisi se bhi \"Gold and Jewelry Shop\" ka pooch sakte hain.", "Shop ka poora naam \"Gold and Jewelry Shop\" hai aur yeh Sarafa Bazar mein waqe hai.", "Aap hamare number 0313******** par WhatsApp kar ke mojudah designs ki pics mangwa sakte hain.",
        "Ji haan! Aap hamare WhatsApp number par message karein, hum aap ko latest designs bhej dein ge.", "Ji haan! Hum designs ke mutabiq asli ya artificial nagine (stones/rubies) bhi lagate hain.", "Ji haan, aap ke order par hum real precious stones (yakoat, zamarrud) jewelry mein laga sakte hain.",
        "Stones aur nagine ki qeemat un ki quality ke hisab se alag se calculate ki jaati hai.", "Sarafa market ka official gold rate aam taur par doper 1 se 2 baje ke darmayan open hota hai.", "Rozana doper ke waqt naya rate market mein aata hai, tab tak pichle din ka rate chalta hai.",
        "Hamari shop se aap ko 100% khalis sona, puka bill, guarantee card aur market se munasib mazdoori milti hai.", "Ahmad Raza sahib pichle keee saalon se imandari se kaam kar rahe hain aur poore Sarafa Bazar mein un ka aitmad hai.", "Hum zyadatar retail customers ko deal karte hain lekin bare orders par wholesale rates ka khayal rakha jata hai.",
        "Yeh aik retail shop hai jahan aam customers ke liye har tarah ki gold jewelry milti hai.", "Jab international market mein gold mehanga hota hai ya dollar ka rate barhta hai, toh Pakistan mein bhi sona mehanga ho jata hai.", "Sone ki qeemat ka daromadar global market par hota hai, is liye is ke sasta hone ka sahi waqt batana mushkil hai.",
        "Ji haan! Hamare paas gold nose pins, laung aur nath ke bohot pyare aur chote designs maujood hain.", "Ji haan, hamari shop par har size ki gold nose pins dastiyab hain.", "Ji haan! Ladies aur bacchon ke liye gold bracelets ke jadeed designs hamari shop par maujood hain.",
        "Ji bilkul, aap shop par aakar hamare designs dekh sakte hain ya apna design bhi laa sakte hain.", "Nahi, hamari shop par sona girwi rakhne ya loan dene ka koi kaam nahi hota, hum sirf khareed-o-froasht karte hain.", "Nahi, hamari shop par udhaar ya girwi rakhne ka system bilkul nahi hai.",
        "Nahi, hamari shop par installment (qiston) par jewelry nahi milti, saari dealing cash ya on-the-spot clear hoti hai.", "Nahi, Ahmad Raza sahib ki shop par qiston (installments) ka koi system nahi hai.", "Hamere paas calcutta design, antique design aur modern light weight jewelry sab se zyada pasand ki jaati hai.",
        "Is waqt market mein light weight laser-cut designs aur antique gold jewelry bohot trend mein hai.", "Hum official sarafa market ka rate follow karte hain. Aap direct hamare number par call kar ke live rate pooch sakte hain.", "Hum All Pakistan Supreme Sarafa Association ka official aur tasdeeq shuda rate follow karte hain.",
        "Ji haan! 22 karat gold mein 91.6% khalis sona hota hai aur baki hissa doosri dhaton ka hota hai taake jewelry mazboot banay.", "24K bilkul 100% pure hota hai jis ki jewelry nahi ban sakti kyunki woh naram hota hai amazon 22K mein 91.6% gold hota hai jo jewelry ke liye perfect hai.", "21 karat gold mein takreeban 87.5% khalis sona maujood hota hai.",
        "Kyunki 18 karat gold mein 75% sona hota hai aur baki doosri dhatein hoti hain, isliye is ka rate 22K ya 24K se kam hota hai.", "Nahi! 18K gold doosri dhaton ki milawat ki wajah se zyada sakht aur mazboot hota hai aur is ka rang bhi kharab nahi hota.", "Ghar par halkay garm pani aur mild soap se naram brush ke zariye saaf kiya ja sakta hai lekin behtar hai aap shop par laakar polish karwayein.",
        "Sona kabhi kharab nahi hota lekin bar bar chemicals ya tez detergent lagane se uski chamak dundli ho sakti hai.", "Ji bilkul! Hamari har raseed par sone ka weight, karat, making charges aur total price wazeh likhi hoti hai.", "Ji haan, hum computerized weigher (kanda) use karte hain aur bilkul accurate weight raseed par darj karte hain.",
        "Hamari shop par bilkul precise aur calibrated computerized weighing scale istemal hota hai jo rati rati ka sahi weight batata hai.", "Hamari shop Sarafa Bazar ke andar waqe hai. Aap market aakar kisi se bhi \"Gold and Jewelry Shop\" ka pooch sakte hain.", "Hamari shop Sarafa Bazar mein waqe hai.",
        "Hamari shop Sarafa Bazar mein hai. Aap wahan aakar hamare number 0313******** par call kar ke mazeed rasta pooch sakte hain.", "Hamari shop Sarafa Bazar mein hai (Aap apne sheher ka naam yahan likh sakte hain).", "Aap Ahmad Raza sahib ki shop par 0313******** par rabta kar sakte hain.",
        "Ji haan! 0313******** par WhatsApp bhi maujood hai aap message kar sakte hain.", "Hamari shop ka number 0313******** hai.", "Aap hamare official number 0313******** par call kar sakte hain.",
        "Shop Monday se Thursday aur Saturday se Sunday tak khuli hoti hai aap in dino mein rabta kar sakte hain.", "Sarafa Bazar mein aam taur par Friday ko weekly off (chutti) hoti hai isliye hamari shop bhi band hoti hai.", "Ji haan! Sunday ko hamari shop open hoti hai aur aap aaram se tashreef laa sakte hain.",
        "Sunday ko bhi shop aam dino ki tarah open hota hai bas Friday ko chutti hoti hai.", "Weekend par Saturday aur Sunday dono din shop open hoti hai sirf Friday ko off hota hai.", "Monday ko shop subah open ho jati hai aur raat tak khuli rehti hai (Friday ke ilawa rozana ka yahi schedule hai).",
        "Ji haan! Monday se Thursday tak shop bilkul khuli hoti hai.", "Agar kal Friday nahi hai toh shop bilkul khuli hogi. Friday ko hamari chutti holidays hoti hai.", "Gold rate din mein kitni baar badalta hai?",
        "Kya aap mujhe abhi ka gold rate bata sakte hain?", "Kya aap log 21 karat gold bechte hain?", "18 karat gold mil jayega?",
        "Karat mein kya farq hota hai?", "Khalis sona kaunsa hota hai?", "Jewelry ke liye sab se behtareen karat kaunsa hai?",
        "Bangles ke designs hain aap ke paas?", "Necklace set banwana ho toh?", "Locket ke designs dekhne hain?",
        "Earrings mil jayenge?", "Kya aap log gents rings banate hain?", "Order par jewelry kitne din mein tayyar hoti hai?",
        "Kya main apni pasand ka design laa sakte hain?", "Agar order emergency mein chahiye ho toh?", "Order dene ke liye advanced payment karni hoti hai?",
        "Kya aap sone ki purity ka certificate dete hain?", "Agar sone mein milaawat ho toh?", "Kya aap purani jewelry ka gold check kar sakte hain?",
        "Karat test karne ke kya charges hain?", "Sone par kaat kitni hoti hai?", "Purana sona exchange karne par naye design ki mazdoori deni hogi?",
        "Kya aap log chandi ke bartan ya jewelry lete hain?", "Payment ke liye cash zaroori hai?", "Kya check accept karte hain?",
        "Dukaan par rash kab hota hai?", "Kya Ahmad Raza sahib se direct baat ho sakti hai?", "Gold and Jewelry Shop kab se bani hui hai?",
        "Aap ki shop ki kya khususiyaat hain?", "Kya aap ke paas choti jewelry bhi hoti hai?", "Karat gold biscuit mil jayenge?",
        "Sona khareedna investment ke liye kaisa hai?"
    ]
}

df = pd.DataFrame(data)

# 2. AI Engine (TF-IDF & Cosine Similarity)
vectorizer = TfidfVectorizer()
questions_matrix = vectorizer.fit_transform(df['Question'].values.astype('U'))

def get_chatbot_response(user_query):
    user_vector = vectorizer.transform([user_query])
    similarities = cosine_similarity(user_vector, questions_matrix)
    best_match_idx = similarities.argmax()
    highest_score = similarities[0][best_match_idx]
    
    if highest_score > 0.22:
        return df['Answer'].iloc[best_match_idx]
    else:
        return "Maazrat, mujhe aap ka sawal samajh nahi aaya. Gold rate, timing, ya shop ke mutabiq kuch aur poochiye."

# 3. Chat History Maintain karne ka system
if "messages" not in st.session_state:
    st.session_state.messages = []

# Purani Chat Display karna
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User ka Input Box (Chat interface)
if user_input := st.chat_input("Apna sawal yahan type karein..."):
    # User ka message screen par show karein
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Bot ka jawab generate karein aur show karein
    response = get_chatbot_response(user_input)
    with st.chat_message("assistant"):
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
