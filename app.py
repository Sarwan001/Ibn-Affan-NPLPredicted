
# ติดตั้งฟอนต์ภาษาไทย
font_path = 'thsarabunnew-webfont.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'TH Sarabun New'
else:
    st.warning("ไม่พบไฟล์ฟอนต์ thsarabunnew-webfont.ttf กราฟอาจแสดงผลเป็นสี่เหลี่ยม")

plt.rcParams['axes.unicode_minus'] = False # ป้องกันเครื่องหมายลบเพี้ยน

# โหลดโมเดลและเครื่องมือ โดยใช้ @st.cache_resource เพื่อให้โหลดครั้งเดียว เว็บจะได้ไม่ช้า
@st.cache_resource
def load_credit_system():
    model = joblib.load('lgb_npl_model.pkl')
    pipeline = joblib.load('credit_preprocessing_pipeline.pkl')
    te_model = pipeline['target_encoder']
    threshold = pipeline['best_threshold']
    return model, te_model, threshold

try:
    model, te_model, threshold = load_credit_system()
except Exception as e:
    st.error(f"ไม่สามารถโหลดไฟล์โมเดลได้ กรุณาตรวจสอบการอัปโหลดไฟล์ (.pkl): {e}")

# ==========================================
# 🔌 ฟังก์ชันโหลดฐานข้อมูลกลุ่มสังกัดจากไฟล์ CSV
# ==========================================
@st.cache_data # บังคับให้โหลดครั้งเดียว เพื่อความรวดเร็วในการเปิดหน้าเว็บ
def load_group_csv():
    path = 'group_master.csv'
    if os.path.exists(path):
        # อ่านไฟล์ CSV พร้อมตั้งค่ารหัสภาษาไทย
        df_group = pd.read_csv(path, encoding='utf-8-sig')
        # ตกแต่งชื่อคอลัมน์ให้เรียบร้อยป้องกันช่องว่างเกิน
        df_group['ชื่อกลุ่ม'] = df_group['ชื่อกลุ่ม'].astype(str).str.strip()

        # 🌟 แปลงโครงสร้างจากตาราง DataFrame ให้เป็น Dictionary เพื่อความรวดเร็วในการค้นหาหลังบ้าน
        # โดยกำหนดให้ 'ชื่อกลุ่ม' เป็น Key หลักของข้อมูล
        database = df_group.set_index('ชื่อกลุ่ม').to_dict('index')
        return database
    else:
        # หากยังหาไฟล์ไม่เจอ ให้คืนค่าดีฟอลต์เปล่าเพื่อไม่ให้แอปพลิเคชันพัง
        return {}

try:
    group_database = load_group_csv()
except Exception as e:
    st.error(f"🚨 ไม่สามารถเชื่อมต่อไฟล์กลุ่มสังกัดได้: {e}")
    group_database = {}

# ตั้งค่าหน้าตาเว็บ
st.set_page_config(page_title="ระบบพิจารณาสินเชื่อ", layout="wide")
st.title("ระบบยื่นคำร้องพิจารณาความเสี่ยงสินเชื่อ")
st.markdown("---")

# แก้ไขบั๊กที่ 1 ปลดล็อกออกจาก st.form เพื่อให้ช่องกรอกข้อมูลกลุ่มตอบสนองทันที (Dynamic)
# และป้องกันไม่ให้กด Enter แล้วระบบยื่นคำร้องทำงานซ้อนกัน

# ---- ส่วนที่ 1: ข้อมูลผู้กู้ ----
st.subheader("ส่วนที่ 1 ข้อมูลส่วนบุคคลผู้ขอสินเชื่อ")
col1, col2, col3 = st.columns(3)
with col1:
    p_gender = st.selectbox("เพศ:", ["ชาย", "หญิง"])
    p_age = st.number_input("อายุผู้กู้ (ปี):", min_value=18, max_value=100, value=35)
with col2:
    p_status = st.selectbox("สถานภาพ:", ["โสด", "สมรส", "หม้าย/หย่าร้าง"])
    p_child = st.number_input("จำนวนบุตร (คน):", min_value=0, max_value=20, value=0)
with col3:
    p_occ = st.selectbox("อาชีพผู้กู้:", [
        "รับจ้าง/ลูกจ้างทั่วไป", "ค้าขาย/ธุรกิจส่วนตัว", "เกษตรกรรม",
        "พนักงานเอกชน/พนักงานบริษัทเอกชน", "ข้าราชการ/รัฐวิสาหกิจ",
        "อาชีพอิสระ/งานบริการ", "ช่าง/รับเหมาก่อสร้าง", "นักศึกษา",
        "ผู้รับบำนาญ", "วิชาชีพเฉพาะ", "ไม่ระบุอาชีพ"
    ])

col1_2, col2_2 = st.columns(2)
with col1_2:
    p_income_type = st.selectbox("ประเภทรายได้ผู้กู้:", ["รายได้ประจำ", "รายได้ไม่ประจำ"])
with col2_2:
    p_income = st.number_input("รายได้ต่อเดือน (บาท):", min_value=0.0, value=25000.0)

st.markdown("---")

# ---- ส่วนที่ 2: ข้อมูลสินเชื่อ ----
st.subheader("ส่วนที่ 2 ข้อมูลส่วนรายละเอียดสินเชื่อ")
col4, col5, col6, col7 = st.columns(4)
with col4:
    l_purpose = st.selectbox("วัตถุประสงค์การกู้:", ["เพื่อการลงทุน", "เพื่อที่อยู่อาศัย", "เพื่อการอุปโภค/บริโภค", "การเกษตรกรรม", "อื่นๆ"])
with col5:
    l_payment_type = st.selectbox("รูปแบบการชำระเงิน:", ["ชำระด้วยตัวเอง", "ฝากผู้แทนฯ", "อื่น ๆ"])
with col6:
    l_amount = st.number_input("วงเงินที่ขอ (บาท):", min_value=0.0, value=50000.0)
with col7:
    l_installment = st.number_input("ชำระต่องวด (บาท):", min_value=0.0, value=30000.0)

st.markdown("---")

# ---- ส่วนที่ 3: ผู้ค้ำและหลักประกัน ----
st.subheader("ส่วนที่ 3 ข้อมูลส่วนหลักประกันและผู้ค้ำประกัน")
g_has_person = st.selectbox("มีบุคคลค้ำประกันหรือไม่?:", ["มี", "ไม่มี"])

# หน้าจอจะเปิดให้กรอกข้อมูลผู้ค้ำเฉพาะตอนเลือก "มี" เท่านั้น
if g_has_person == "มี":
    col8, col9, col10 = st.columns(3)
    with col8:
        g_age = st.number_input("อายุผู้ค้ำ (ปี):", min_value=18, max_value=100, value=40)
    with col9:
        g_occ = st.selectbox("อาชีพผู้ค้ำ:", [
            "รับจ้าง/ลูกจ้างทั่วไป", "ค้าขาย/ธุรกิจส่วนตัว", "เกษตรกรรม",
            "พนักงานเอกชน/พนักงานบริษัทเอกชน", "ข้าราชการ/รัฐวิสาหกิจ",
            "อาชีพอิสระ/งานบริการ", "ช่าง/รับเหมาก่อสร้าง", "นักศึกษา",
            "ผู้รับบำนาญ", "วิชาชีพเฉพาะ", "ไม่ระบุอาชีพ"
        ])
    with col10:
        g_income_type = st.selectbox("ประเภทรายได้ผู้ค้ำ:", ["รายได้ประจำ", "รายได้ไม่ประจำ"])
else:
    g_age = 0
    g_occ = "ไม่ระบุอาชีพ"
    g_income_type = "รายได้ไม่ประจำ"
    st.caption("ระบบเซ็ตค่าผู้ค้ำเป็นค่าเริ่มต้น (0) ให้อัตโนมัติ")

col11, col12 = st.columns(2)
with col11:
    c_has_land = st.selectbox("มีหลักค้ำโฉนดหรือไม่?:", ["มี", "ไม่มี"])
    c_land_val = st.number_input("มูลค่าโฉนดเอกชน (บาท):", min_value=0.0, value=0.0) if c_has_land == "มี" else 0.0
with col12:
    c_has_deposit = st.selectbox("มีหลักค้ำเงินฝากหรือไม่?:", ["มี", "ไม่มี"])
    c_deposit_val = st.number_input("จำนวนเงินค้ำเงินฝาก (บาท):", min_value=0.0, value=0.0) if c_has_deposit == "มี" else 0.0

st.markdown("---")

# ---- ส่วนที่ 4: ข้อมูลกลุ่มสังกัด (ฉบับยืดหยุ่นผ่านไฟล์ CSV) ----
st.subheader("ส่วนที่ 4 ข้อมูลกลุ่มสังกัด")

# ดึงรายชื่อคีย์ทั้งหมดจากไฟล์ CSV มาร้อยเรียงเป็นตัวเลือก Dropdown
group_options = ["ไม่มี", "สมาชิกสหกรณ์ทั่วไป"] + list(group_database.keys())
grp_name = st.selectbox("เลือกระบุชื่อกลุ่มสังกัด:", group_options)

if grp_name in ["ไม่มี", "สมาชิกสหกรณ์ทั่วไป"]:
    st.caption(f"ไม่มีกลุ่มสังกัด {grp_name}")
    grp_age = 0
    grp_occ = "ไม่ระบุอาชีพ"
    grp_income_type = "รายได้ไม่ประจำ"
    grp_income = 0.0
else:
    # ระบบวิ่งไปจับคู่ข้อมูลจากชื่อกลุ่มในไฟล์ CSV มาใส่ตัวแปรให้โมเดลอัตโนมัติ
    leader_info = group_database[grp_name]

    # ดึงค่าตามชื่อคอลัมน์ที่คุณเซฟไว้ในไฟล์ CSV
    grp_age = leader_info.get("อายุหัวหน้ากลุ่ม", 0)
    grp_occ = leader_info.get("อาชีพหัวหน้ากลุ่ม", "ไม่ระบุอาชีพ")
    grp_income_type = leader_info.get("ประเภทรายได้หัวหน้ากลุ่ม", "รายได้ไม่ประจำ")
    grp_income = leader_info.get("รายได้หัวหน้ากลุ่ม", 0.0)

    # แสดงแผงข้อมูล (Metrics Dashboard) ให้เจ้าหน้าที่ตรวจสอบความถูกต้อง
    st.success(f"ข้อมูลส่วนบุคคลหัวหน้ากลุ่ม : {grp_name} ")
    with st.container():
        col13, col14, col15 = st.columns(3)
        with col13:
            st.metric("อายุหัวหน้ากลุ่ม", f"{grp_age} ปี")
            st.metric("รายได้หัวหน้ากลุ่ม", f"{grp_income:,.0f} บาท")
        with col14:
            st.metric("อาชีพหัวหน้ากลุ่ม", grp_occ)
        with col15:
            st.metric("ประเภทรายได้", grp_income_type)

st.markdown("---")
# ใช้ปุ่มกดเดี่ยวๆ เพื่อแยกฟังก์ชันการทำงานอย่างชัดเจน
submit_btn = st.button("ยืนยันข้อมูล/ยื่นคำร้องพิจารณาสินเชื่อ")

# ส่วนประมวลผลหลังบ้านเมื่อกดปุ่มยื่นขอสินเชื่อ
if submit_btn:
    raw_data_dict = {
        'เพศ': p_gender, 'อายุผู้กู้': p_age, 'สถานภาพ': p_status, 'จำนวนบุตร': p_child,
        'อาชีพ': p_occ, 'ประเภทรายได้': p_income_type, 'รายได้': p_income,
        'วัตถุประสงค์': l_purpose, 'รูปแบบการชำระ': l_payment_type, 'วงเงินที่ขอ': l_amount, 'ชำระต่องวด': l_installment,
        'หลักค้ำบุคคล': g_has_person, 'อายุผู้ค้ำ': g_age, 'อาชีพผู้ค้ำ': g_occ, 'ประเภทรายได้ผู้ค้ำ': g_income_type,
        'หลักค้ำโฉนด': c_has_land, 'มูลค่าโฉนดเอกชน': c_land_val, 'หลักค้ำเงินฝาก': c_has_deposit, 'จำนวนเงินค้ำเงินฝาก': c_deposit_val,
        'ชื่อกลุ่ม': grp_name, 'อายุหัวหน้ากลุ่ม': grp_age, 'อาชีพหัวหน้ากลุ่ม': grp_occ,
        'ประเภทรายได้หัวหน้ากลุ่ม': grp_income_type, 'รายได้หัวหน้ากลุ่ม': grp_income
    }

    df = pd.DataFrame([raw_data_dict])

    # Feature Engineering
    df['เพศ'] = np.where(df['เพศ'] == 'ชาย', 1, 0)
    df['ประเภทรายได้'] = np.where(df['ประเภทรายได้'] == 'รายได้ประจำ', 1, 0)
    df['ประเภทรายได้ผู้ค้ำ'] = np.where(df['ประเภทรายได้ผู้ค้ำ'] == 'รายได้ประจำ', 1, 0)
    df['ประเภทรายได้หัวหน้ากลุ่ม'] = np.where(df['ประเภทรายได้หัวหน้ากลุ่ม'] == 'รายได้ประจำ', 1, 0)
    df['หลักค้ำบุคคล'] = np.where(df['หลักค้ำบุคคล'] == 'มี', 1, 0)
    df['หลักค้ำโฉนด'] = np.where(df['หลักค้ำโฉนด'] == 'มี', 1, 0)
    df['หลักค้ำเงินฝาก'] = np.where(df['หลักค้ำเงินฝาก'] == 'มี', 1, 0)

    df['รายได้หลังหักหนี้'] = df['รายได้'] - df['ชำระต่องวด']
    df['Cov_Ratio'] = np.where(df['ชำระต่องวด'] > 0, df['รายได้หลังหักหนี้'] / df['ชำระต่องวด'], 0)
    df['DTI'] = np.where(df['รายได้'] > 0, (df['ชำระต่องวด'] / df['รายได้']) * 100, 0)

    cond3 = (df['มูลค่าโฉนดเอกชน'] == 0) & (df['หลักค้ำเงินฝาก'] == 1) & (df['หลักค้ำบุคคล'] == 1)
    cond2_1 = (df['มูลค่าโฉนดเอกชน'] != 0) & (df['หลักค้ำเงินฝาก'] == 0) & (df['หลักค้ำบุคคล'] == 1)
    cond2_2 = (df['หลักค้ำเงินฝาก'] == 1) & (df['มูลค่าโฉนดเอกชน'] == 0) & (df['หลักค้ำบุคคล'] == 1)
    cond4 = (df['มูลค่าโฉนดเอกชน'] == 0) & (df['หลักค้ำเงินฝาก'] == 0) & (df['หลักค้ำบุคคล'] == 1)
    choices = [30000, 30000, 30000, df['วงเงินที่ขอ'] + 10000]
    df['วงเงินบุคคลค้ำ'] = np.select([cond3, cond2_1, cond2_2, cond4], choices, default=0)
    df['รวมมูลค่าสินทรัพย์ค้ำประกัน'] = df['วงเงินบุคคลค้ำ'] + df['จำนวนเงินค้ำเงินฝาก'] + df['มูลค่าโฉนดเอกชน']
    df['LTV'] = np.where(df['รวมมูลค่าสินทรัพย์ค้ำประกัน'] > 0, (df['วงเงินที่ขอ'] / df['รวมมูลค่าสินทรัพย์ค้ำประกัน']) * 100, 0)

    # ส่วนแสดงผล Feature Engineering
    st.markdown("## ข้อมูลเชิงลึกสุขภาพทางการเงิน")
    st.info("ผลการประเมินสุขภาพทางการเงินเบื้องต้น")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.metric(label="รายได้คงเหลือหลังหักหนี้", value=f"{df['รายได้หลังหักหนี้'][0]:,.2f} บาท")
        # แสดง DTI พร้อมเตือนถ้าเกิน 40%
        dti_val = df['DTI'][0]
        st.metric(label="สัดส่วนภาระหนี้ต่อรายได้ DTI", value=f"{dti_val:,.2f}%",
                  delta="⚠️ มีภาระหนี้สูง" if dti_val > 40 else "มีภาระหนี้ต่ำ", delta_color="inverse")
    with col_f2:
        st.metric(label="อัตราความสามารถในการชำระหนี้ Coverage Ratio", value=f"{df['Cov_Ratio'][0]:,.2f} เท่า")
        st.metric(label="มูลค่าสินทรัพย์หลักประกันรวม", value=f"{df['รวมมูลค่าสินทรัพย์ค้ำประกัน'][0]:,.2f} บาท")
    with col_f3:
        ltv_val = df['LTV'][0]
        st.metric(label="สัดส่วนวงเงินกู้ต่อหลักประกัน LTV", value=f"{ltv_val:,.2f}%",
                  delta="⚠️ สัดส่วนหลักประกันมีความเสี่ยง" if ltv_val > 80 else "หลักประกันสามารถครอบคลุมหนี้", delta_color="inverse")

    st.markdown("---")

    # One-Hot Encoding Manual
    df['สถานภาพ_สมรส'] = np.where(df['สถานภาพ'] == 'สมรส', 1, 0)
    df['สถานภาพ_หม้าย/หย่าร้าง'] = np.where(df['สถานภาพ'] == 'หม้าย/หย่าร้าง', 1, 0)
    df['สถานภาพ_โสด'] = np.where(df['สถานภาพ'] == 'โสด', 1, 0)
    df['รูปแบบการชำระ_ชำระด้วยตัวเอง'] = np.where(df['รูปแบบการชำระ'] == 'ชำระด้วยตัวเอง', 1, 0)
    df['รูปแบบการทำชำระ_ฝากผู้แทนฯ'] = np.where(df['รูปแบบการชำระ'] == 'ฝากผู้แทนฯ', 1, 0)
    df['รูปแบบการชำระ_อื่น ๆ'] = np.where(df['รูปแบบการชำระ'] == 'อื่น ๆ', 1, 0)
    df['วัตถุประสงค์_เพื่อการลงทุน'] = np.where(df['วัตถุประสงค์'] == 'เพื่อการลงทุน', 1, 0)
    df['วัตถุประสงค์_เพื่อที่อยู่อาศัย'] = np.where(df['วัตถุประสงค์'] == 'เพื่อที่อยู่อาศัย', 1, 0)
    df['วัตถุประสงค์_เพื่อการอุปโภค/บริโภค'] = np.where(df['วัตถุประสงค์'] == 'เพื่อการอุปโภค/บริโภค', 1, 0)
    df['วัตถุประสงค์_การเกษตรกรรม'] = np.where(df['วัตถุประสงค์'] == 'การเกษตรกรรม', 1, 0)
    df['วัตถุประสงค์_อื่นๆ'] = np.where(df['วัตถุประสงค์'] == 'อื่นๆ', 1, 0)

    final_expected_columns = [
        'วงเงินที่ขอ', 'ชำระต่องวด', 'Cov_Ratio', 'เพศ', 'อาชีพ', 'ประเภทรายได้',
        'รายได้', 'จำนวนบุตร', 'อายุผู้กู้', 'หลักค้ำบุคคล', 'วงเงินบุคคลค้ำ',
        'อาชีพผู้ค้ำ', 'อายุผู้ค้ำ', 'หลักค้ำโฉนด', 'มูลค่าโฉนดเอกชน', 'หลักค้ำเงินฝาก',
        'จำนวนเงินค้ำเงินฝาก', 'LTV', 'ชื่อกลุ่ม', 'อาชีพหัวหน้ากลุ่ม', 'รายได้หัวหน้ากลุ่ม',
        'อายุหัวหน้ากลุ่ม', 'ประเภทรายได้หัวหน้ากลุ่ม', 'ประเภทรายได้ผู้ค้ำ',
        'รายได้หลังหักหนี้', 'DTI', 'รวมมูลค่าสินทรัพย์ค้ำประกัน',
        'รูปแบบการชำระ_ชำระด้วยตัวเอง', 'รูปแบบการทำชำระ_ฝากผู้แทนฯ', 'รูปแบบการชำระ_อื่น ๆ',
        'สถานภาพ_สมรส', 'สถานภาพ_หม้าย/หย่าร้าง', 'สถานภาพ_โสด',
        'วัตถุประสงค์_เพื่อการลงทุน', 'วัตถุประสงค์_เพื่อที่อยู่อาศัย',
        'วัตถุประสงค์_เพื่อการอุปโภค/บริโภค', 'วัตถุประสงค์_การเกษตรกรรม', 'วัตถุประสงค์_อื่นๆ'
    ]

    # แก้ไขบั๊กที่ 2: ลบโค้ดชุดแปลงข้อมูล 4 คอลัมน์เดิมที่ทำให้มิติพังทลายออกไป
    # แล้วใช้โครงสร้างกางรับ 38 คอลัมน์ให้พร้อมก่อนส่งเข้าแปลงในคำสั่ง te_modelด้านล่างนี้
    X_ready = df.reindex(columns=final_expected_columns, fill_value=0)

    # ส่งตารางที่มีโครงสร้าง 38 คอลัมน์ที่ถูกต้องเข้าไปทำ Target Encoding รอบเดียวจบ!
    X_ready = te_model.transform(X_ready)

    # ---- 4. ทำนายผลความเสี่ยง ----
    pd_score = model.predict_proba(X_ready)[:, 1][0]

    st.markdown("ผลการประเมินความเสี่ยงสินเชื่อ")
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric(label="โอกาสเสี่ยงที่จะเกิดหนี้เสีย ", value=f"{pd_score * 100:.2f}%")
    with col_res2:
        if pd_score >= threshold:
            st.error(f"ไม่อนุมัติ เนื่องจากมีความเสี่ยงเกินเกณฑ์ที่กำหนด {threshold*100:.2f}%")
        else:
            st.success(f"อนุมัติสินเชื่อ เนื่องจากมีความเสี่ยงอยู่ในเกณฑ์ที่ปลอดภัย")

    # ---- 5. แสดงสเต็ปคณิตศาสตร์อธิบายผู้ใช้ ----
    st.markdown("---")
    st.subheader("ขั้นตอนการคำนวณอัตราความเสี่ยง")
    explainer_lgb = shap.TreeExplainer(model)
    expected_value_raw = explainer_lgb.expected_value
    shap_vals_raw = explainer_lgb.shap_values(X_ready)

    base_val = expected_value_raw[1] if isinstance(expected_value_raw, (list, np.ndarray)) and len(expected_value_raw) > 1 else expected_value_raw[0] if isinstance(expected_value_raw, (list, np.ndarray)) else expected_value_raw

    if isinstance(shap_vals_raw, list): shap_vals = shap_vals_raw[1] if len(shap_vals_raw) > 1 else shap_vals_raw[0]
    elif isinstance(shap_vals_raw, np.ndarray) and len(shap_vals_raw.shape) == 3: shap_vals = shap_vals_raw[:, :, 1] if shap_vals_raw.shape[2] > 1 else shap_vals_raw[:, :, 0]
    else: shap_vals = shap_vals_raw

    final_log_odds = base_val + np.sum(shap_vals[0])

    # แสดงสเต็ปเลขคณิต
    st.code(f"""
1. คะแนนความเสี่ยงจากกราฟน้ำตก f(x) = {final_log_odds:.4f}
2. คำนวณผ่านฟังก์ชัน Sigmoid function :
   P = 1 / (1 + e^-f(x))
   P = 1 / (1 + e^(-({final_log_odds:.4f})))
   P = 1 / (1 + {np.exp(-final_log_odds):.4f})
   P = {pd_score:.4f}
3. อัตราความเสี่ยงที่จะเป็น NPL = {pd_score * 100:.2f}%
    """)

    # ---- 6. วาดกราฟ SHAP แสดงบนหน้าเว็บ ----
    st.subheader("การวิเคราะห์ปัจจัยเชิงลึกด้วย SHAP Value")

    shap_expl_lgb = shap.Explanation(
        values=shap_vals[0], base_values=base_val, data=X_ready.iloc[0].values, feature_names=X_ready.columns.tolist()
    )

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_expl_lgb, max_display=10, show=False)
    plt.title("เจาะลึกปัจจัยที่มีอิทธิพลต่อความเสี่ยงของสมาชิก", fontsize=12, pad=15)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    shap.plots.bar(shap_expl_lgb, max_display=10, show=False)
    plt.title("10 อันดับปัจจัยที่ส่งผลต่อความเสี่ยงของสมาชิก", fontsize=12, pad=15)
    st.pyplot(fig2)
