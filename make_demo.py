import cv2, numpy as np, json
W,H=760,520
def draw(underline=True):
    im=np.full((H,W,3),248,np.uint8); F=cv2.FONT_HERSHEY_SIMPLEX
    cv2.ellipse(im,(190,70),(46,30),0,0,360,(20,20,20),12)
    cv2.putText(im,"AGC AUTOMOTIVE",(70,150),F,0.95,(20,20,20),2,cv2.LINE_AA)
    cv2.circle(im,(95,205),26,(20,20,20),3); cv2.putText(im,"E6",(78,214),F,0.7,(20,20,20),2,cv2.LINE_AA)
    cv2.putText(im,"43R-008574",(140,215),F,0.95,(20,20,20),2,cv2.LINE_AA)
    cv2.putText(im,"M1H3S",(330,285),F,0.85,(20,20,20),2,cv2.LINE_AA)
    cv2.circle(im,(95,330),26,(20,20,20),3)
    cv2.putText(im,"TIS 2602-2556",(140,340),F,0.85,(20,20,20),2,cv2.LINE_AA)
    cv2.putText(im,"T",(128,372),F,0.7,(20,20,20),2,cv2.LINE_AA)
    cv2.putText(im,"11T",(400,372),F,0.7,(20,20,20),2,cv2.LINE_AA)
    cv2.putText(im,"T1",(40,420),F,0.9,(20,20,20),2,cv2.LINE_AA)
    cv2.putText(im,"5NI",(105,420),F,0.9,(20,20,20),2,cv2.LINE_AA)
    if underline: cv2.line(im,(103,393),(165,393),(20,20,20),4)   # <-- ขีดบน/ใต้ 5NI
    cv2.putText(im,"TEMPERLITE",(200,420),F,0.9,(20,20,20),2,cv2.LINE_AA)
    cv2.putText(im,"2   4",(210,470),F,0.8,(20,20,20),2,cv2.LINE_AA)
    return im
master=draw(True); cv2.imwrite("master.jpg",master)

# --- สร้าง test: ขีดหาย + เอียง + แสงไม่สม่ำเสมอ + noise + ปากกาน้ำเงิน
t=draw(False)
cv2.line(t,(300,60),(420,120),(150,40,20),4)      # ปากกาน้ำเงิน (BGR)
cv2.line(t,(430,200),(560,250),(160,50,25),4)
cv2.putText(t,"OK",(560,430),cv2.FONT_HERSHEY_SIMPLEX,1.4,(150,40,20),4)
src=np.float32([[0,0],[W,0],[W,H],[0,H]]); dst=np.float32([[25,12],[W-8,30],[W-30,H-6],[10,H-25]])
t=cv2.warpPerspective(t,cv2.getPerspectiveTransform(src,dst),(W,H),borderValue=(248,248,248))
g=np.tile(np.linspace(0.75,1.12,W),(H,1))[...,None]           # ไล่แสง
t=np.clip(t*g,0,255).astype(np.uint8)
t=np.clip(t.astype(np.float32)+np.random.normal(0,5,t.shape),0,255).astype(np.uint8)
cv2.imwrite("test_ng.jpg",t)

# test OK (ขีดครบ) ผ่านการบิดเบือนเหมือนกัน
o=draw(True); cv2.line(o,(300,60),(420,120),(150,40,20),4)
o=cv2.warpPerspective(o,cv2.getPerspectiveTransform(src,dst),(W,H),borderValue=(248,248,248))
o=np.clip(o*g,0,255).astype(np.uint8)
cv2.imwrite("test_ok.jpg",o)

json.dump({"master":"master.jpg","rois":[
 {"name":"5NI_UNDERLINE","x":90,"y":378,"w":90,"h":26,"mode":"line","line_min_len_ratio":0.6},
 {"name":"5NI_TEXT","x":98,"y":398,"w":80,"h":36,"mode":"ink","min_ratio":0.55,"min_corr":0.4},
 {"name":"TEMPERLITE","x":195,"y":395,"w":260,"h":40,"mode":"ink","min_ratio":0.55,"min_corr":0.4},
 {"name":"E6_APPROVAL","x":60,"y":180,"w":330,"h":60,"mode":"ink","min_ratio":0.55,"min_corr":0.4}
]},open("roi.json","w"),indent=2)
print("demo created")
