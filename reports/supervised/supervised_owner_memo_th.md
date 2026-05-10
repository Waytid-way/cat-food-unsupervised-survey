# บันทึกสำหรับทีมธุรกิจ

- โมเดลที่ใช้เป็น source of truth: `random_forest`
- Accuracy บน holdout set: 0.880
- Macro F1 บน holdout set: 0.797
- ใช้ dashboard แท็บ Supervised เพื่อดูการเทียบโมเดล, confusion matrix, และ feature importance
- ถ้าต้องการ export ไปใช้งานภายนอก ให้หยิบ `best_model.pkl` และ `metrics_summary.json` เป็นหลัก

## วิธีตีความผล
- ถ้า confusion matrix มีค่าผิดพลาดสูงในบางกลุ่ม แปลว่า segment อาจยังซ้อนทับกัน และควรทบทวน feature engineering
- ถ้า feature importance กระจุกที่คอลัมน์เดิมจำนวนมาก แปลว่าข้อมูล survey มี signal ค่อนข้างแคบ ควรพิจารณาเพิ่มคำถามในรอบถัดไป
