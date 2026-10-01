import cv2, numpy as np, json
W,H=900,640
F=cv2.FONT_HERSHEY_SIMPLEX
def base():
    im=np.full((H,W,3),250,np.uint8)
    cv2.ellipse(im,(250,70),(55,34),0,0,360,(15,15,15),14)          # logo
    cv2.putText(im,"AGC AUTOMOTIVE",(80,170),F,1.3,(15,15,15),3,cv2.LINE_AA)
    cv2.circle(im,(120,250),40,(15,15,15),4)                         # RING E6
    cv2.putText(im,"E6",(95,265),F,1.0,(15,15,15),3,cv2.LINE_AA)
    cv2.putText(im,"43R-008574",(185,265),F,1.3,(15,15,15),3,cv2.LINE_AA)
    cv2.putText(im,"M1H3S",(430,350),F,1.2,(15,15,15),3,cv2.LINE_AA)
    cv2.circle(im,(120,430),42,(15,15,15),4)                         # RING TIS
    cv2.putText(im,"TIS 2602-2556",(185,445),F,1.2,(15,15,15),3,cv2.LINE_AA)
    cv2.putText(im,"T",(178,495),F,0.9,(15,15,15),3,cv2.LINE_AA)
    cv2.putText(im,"11T",(560,495),F,0.9,(15,15,15),3,cv2.LINE_AA)
    cv2.putText(im,"T1",(40,555),F,1.2,(15,15,15),3,cv2.LINE_AA)
    cv2.putText(im,"SNI",(125,555),F,1.1,(15,15,15),3,cv2.LINE_AA)
    cv2.line(im,(122,520),(215,520),(15,15,15),5)                    # ขีดบน SNI
    cv2.putText(im,"TEMPERLITE",(255,555),F,1.2,(15,15,15),3,cv2.LINE_AA)
    cv2.putText(im,"2    4    6    8",(300,615),F,1.0,(15,15,15),3,cv2.LINE_AA)  # ignore zone
    return im

m=base(); cv2.imwrite("m2.png",m)

# ---- NG: 1) ขีดบน SNI หาย 2) 'M' ใน M1H3S หายครึ่งซ้าย 3) วงกลม TIS หายครึ่งล่าง
#          4) '4' ใน 43R หายทั้งตัว 5) เลข 2468 เปลี่ยนไป (ต้องถูก ignore)
t=base()
cv2.line(t,(122,520),(215,520),(250,250,250),9)                    # ลบขีด
cv2.rectangle(t,(428,320),(452,365),(250,250,250),-1)              # ครึ่งซ้าย M
cv2.ellipse(t,(120,430),(46,46),0,20,160,(250,250,250),10)         # ครึ่งล่างวงกลม TIS
cv2.rectangle(t,(183,235),(212,272),(250,250,250),-1)              # ลบเลข 4
cv2.rectangle(t,(295,585),(620,625),(250,250,250),-1)              # ลบ 2468 ทั้งแถว
cv2.putText(t,"9    9",(300,615),F,1.0,(15,15,15),3,cv2.LINE_AA)   # ใส่เลขมั่วแทน
# บิด + แสง + ปากกา
cv2.line(t,(650,120),(820,200),(150,40,20),5)
src=np.float32([[0,0],[W,0],[W,H],[0,H]]); dst=np.float32([[22,10],[W-10,26],[W-26,H-8],[8,H-22]])
t=cv2.warpPerspective(t,cv2.getPerspectiveTransform(src,dst),(W,H),borderValue=(250,250,250))
g=np.tile(np.linspace(0.78,1.10,W),(H,1))[...,None]
t=np.clip(t.astype(np.float32)*g+np.random.normal(0,4,(H,W,3)),0,255).astype(np.uint8)
cv2.imwrite("t2_ng.png",t)

# OK case
o=base(); cv2.rectangle(o,(295,585),(620,625),(250,250,250),-1)
cv2.putText(o,"1    3    5",(300,615),F,1.0,(15,15,15),3,cv2.LINE_AA)  # เลขต่าง แต่ต้อง ignore
cv2.line(o,(650,120),(820,200),(150,40,20),5)
o=cv2.warpPerspective(o,cv2.getPerspectiveTransform(src,dst),(W,H),borderValue=(250,250,250))
o=np.clip(o.astype(np.float32)*g,0,255).astype(np.uint8)
cv2.imwrite("t2_ok.png",o)

json.dump({"ignore":[{"name":"DOT_2468","x":270,"y":578,"w":380,"h":58}]},
          open("ignore.json","w"),indent=2)
print("demo2 created")
