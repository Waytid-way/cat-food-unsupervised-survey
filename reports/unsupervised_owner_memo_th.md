# Unsupervised Owner Memo

## สิ่งที่ต้องจำตอนพรีเซนต์
- ใช้ data หลังกรองจริง 148 คน จาก raw 167 คน
- จุดขายของงานนี้คือการใช้ ipsatization เพื่อลด response style bias ก่อนทำ PCA
- ผลสุดท้ายเลือก `k = 2` เพราะ silhouette ดีสุดในชุดทดสอบและอธิบายง่าย
- hierarchical validation ให้ cophenetic correlation = 0.473 ซึ่งบอกว่าโครงสร้างกลุ่มมี แต่ไม่คมมาก
- anomaly detection เจอ 26 คน (17.6%)

## ถ้าอาจารย์ถามว่าทำไมไม่ใช้ k=3
- ตอบว่า dataset จริงมี overlap สูง และ silhouette ของ k=2 ดีกว่าชุดอื่นในรอบ sweep
- k=2 ให้กลุ่มที่อธิบายการตลาดได้ง่ายกว่า ขณะที่ k ที่มากขึ้นแยกย่อยแต่ไม่ได้เพิ่มคุณภาพเชิง metric ชัดพอ

## ถ้าอาจารย์ถามว่าทำไมต้อง ipsatize
- เพราะผู้ตอบบางคนมีแนวโน้มให้คะแนนสูงทุกข้อ
- การลบค่าเฉลี่ยรายแถวช่วยดึง preference จริงของแพ็กเกจออกจาก response style bias

## โฟกัสเชิงธุรกิจ
- กลุ่มที่เล็กกว่าเป็นกลุ่มที่เข้มกับคุณภาพและดีไซน์ ควรใช้ messaging แบบ premium
- กลุ่มใหญ่กว่าเป็นฐานตลาดหลัก ควรเน้นความชัดเจน ความคุ้มค่า และข้อมูลที่เข้าใจง่าย

## ไฟล์ที่ส่งต่อ
- `clean_dataset_with_segments.csv`
- `metrics_summary.json`
- `correlation_matrix.csv`
- `segment_profiles.csv`
