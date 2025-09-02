# -*- coding: utf-8 -*-
"""
Bu sürümde (1) SON WPT'YE VARINCA GERÇEKTEN DUR/BEKLE (COLOR WAIT) mantığı eklendi,
(2) MAVLink'ten renk kodu dinleme (STATUSTEXT / NAMED_VALUE_INT / PARAM_VALUE) geldi,
(3) Son WPT'de pozisyon tutma için aynı WPT'ye tekrar tekrar konum komutu ve düşük hız,
(4) HUD/Log zenginleştirildi

NOT:
- “5. nokta” son nokta olarak kabul edilir. Aracın burada durup beklemesi hedeflenir.
- Renk kodu gelmediği sürece burada bekler (timeout opsiyonel).
- Bu eklemeler mevcut NAV/kaçınma mimarisini bozmaz; yalnızca son WPT logic'ine
  bir bekleme katmanı (colorwait) ekler.
"""

import cv2
import numpy as np
import gi
from ultralytics import YOLO
from pymavlink import mavutil
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import math
import time

cv2.setUseOptimized(True)

# =======================
# PARAMETRELER
# =======================
CONF_THR = 0.30
ALPHA_AVOID      = 0.75
MAX_AVOID_DEG    = 60.0
LOOKAHEAD_BASE_M = 5.0
LOOKAHEAD_MIN_M  = 3.0
LOOKAHEAD_MAX_M  = 8.0

# --- hızlar ---
CRUISE_SPEED_MPS       = 2.0
AVOID_SPEED_MPS        = 0.70
PREEMPT_SPEED_MPS      = 0.45
EMERG_SPEED_MPS        = 0.16
MIN_CRUISE_FLOOR_MPS   = 1.20
MIN_AVOID_FLOOR_MPS    = 0.80
SPEED_CMD_REFRESH_SEC  = 1.5

# --- görüş / duba ölçüleri ---
CAMERA_VFOV_DEG   = 60.0
CAMERA_HFOV_DEG   = 60.0
BUOY_HEIGHT_M     = 1.20
BUOY_WIDTH_M      = 0.35
DEAD_AHEAD_BAND   = 0.18

# --- PREEMPT / EMERG ---
PREEMPT_TRIG_DIST_M  = 9.0
PREEMPT_CLEAR_DIST_M = 8.0
PREEMPT_TTC_S        = 6.0
PREEMPT_BIAS_DEG     = 16.0
PREEMPT_COMMIT_S     = 2.0

FRONT_TRIG_DIST_M    = 7.5
FRONT_CLEAR_DIST_M   = 6.5
EMERG_TTC_S          = 3.5
FRONT_BIAS_DEG       = 38.0
EMERG_COMMIT_S       = 2.6   # (talep edilen değer korunur)
FRONTAL_PUSH_MIN_DEG = 28.0

STOP_TTC_S           = 1.7
STOP_SPEED_MPS       = 0.08   # tam duruşa yakın düşük hız

# --- PANIC ---
PANIC_TRIG_DIST_M    = 4.2
PANIC_TTC_S          = 2.2
PANIC_BIAS_DEG       = 45.0
PANIC_SPEED_MPS      = 0.06
PANIC_COMMIT_S       = 1.2

# --- ÇATIŞMA ---
CONFLICT_AREA_THR        = 0.012
CONFLICT_NEAR_DIST_M     = 7.0
CONFLICT_DIST_MARGIN_M   = 1.0
CONFLICT_BIAS_DEG        = 14.0
CONFLICT_COMMIT_S        = 1.6
CONFLICT_SPEED_MPS       = 0.85

# --- RECOVERY ---
RECOVERY_DIST_RESUME_M   = 1.8
RECOVERY_TIME_MAX_S      = 3.0
RESUME_FRONT_CLEAR_M     = 6.0
RESUME_TTC_CLEAR_S       = 4.0
RECOVERY_BIAS_DEG        = 20.0
RECOVERY_SPEED_MPS       = 0.60
RECOVERY_LOOKAHEAD_M     = 3.0

# TTC filtresi
TTC_EMA_ALPHA        = 0.6
TTC_FORGET_SEC       = 0.8

# Yakınlık heuristikleri
NEAR_PIX_BOTTOM_FRAC = 0.86
AREA_NEAR_FRAC       = 0.022

# renk ağırlıkları
AVOID_COLOR_SET = {"yellow", "orange"}
COLOR_GAIN      = 1.0
NONTARGET_GAIN  = 0.50
UNKNOWN_GAIN    = 0.70

CENTER_BAND      = 0.45
CENTER_BOOST_DEG = 30.0

# --- HAT TAKİBİ / DAR KÖŞE ---
LINE_TRACK_ENABLE   = True
STANLEY_K           = 0.75
STANLEY_MAX_DEG     = 22.0
LINE_USE_THRESH_DEG = 6.0
CORNER_LOCK_S       = 1.60
CORNER_LOOKAHEAD_M  = 2.2
CORNER_EXTRA_BIAS   = 8.0

# --- YAKIN-WP KORUMALI (Guard) ---
NEAR_WP_R_M            = 12.0
GUARD_OFFSET_M         = 2.6
GUARD_COMMIT_S         = 1.6
GUARD_LOOKAHEAD_M      = 2.6
GUARD_SPEED_MPS        = 0.60
ALPHA_AVOID_NEAR_GAIN  = 1.15

# --- STUCK + BYPASS (kama) ---
STUCK_WINDOW_S        = 10.0
STUCK_MOVE_THRESH_M   = 1.5
STUCK_FLIP_WINDOW_S   = 6.0
STUCK_FLIP_COUNT      = 4

BYPASS_NEAR_WP_R_M    = 15.0
BYPASS_ANGLE_DEG      = 80.0
BYPASS_LOOKAHEAD_M    = 2.4
BYPASS_SPEED_MPS      = 0.55
BYPASS_COMMIT_S       = 3.0
BYPASS_MIN_TRAVEL_M   = 2.0
BYPASS_ALPHA_GAIN     = 1.25
BYPASS_EXTRA_BIAS     = 6.0

# --- SON WPT RENK-BEKLEME (COLOR WAIT) ---
COLOR_WAIT_ENABLE     = True           # Son waypoint'e VARINCA dur ve bekle
COLOR_WAIT_TIMEOUT_S  = None           # Örn. 90.0 yaparsan o kadar saniye sonra hala bekler (hareket yok)
# Kabul edilecek MAVLink kaynakları:
# 1) STATUSTEXT: "COLOR: RED" / "RENK: KIRMIZI"
# 2) NAMED_VALUE_INT: name ∈ {"COLOR","RENK","COLOR_ID","RENK_ID"} ; value ∈ {0..4}
# 3) PARAM_VALUE:     param_id aynı isimler; param_value ∈ {0..4}

# =======================
# WPT (doğrudan lat/lon)
# =======================
lat0 = 51.566151
lon0 = -4.034345
alt0 = 10.0

coordinates = []
targetCoordinate = 0

def AddWaypointDeg(lat_deg, lon_deg, alt):
    coordinates.append((float(lat_deg), float(lon_deg), float(alt)))
    print("[WPT ADDED DEG]", (lat_deg, lon_deg, alt))

AddWaypointDeg(51.566151,            -4.034282858920814, 10.0)
AddWaypointDeg(51.566227356799146,   -4.034402805655057, 10.0)
AddWaypointDeg(51.566294730445456,   -4.034330548586236, 10.0)
AddWaypointDeg(51.566366595668185,   -4.034366677120646, 10.0)
AddWaypointDeg(51.56672592178183,    -4.034345,          10.0)  # 5. ve SON nokta

# =======================
# GSTREAMER
# =======================
Gst.init(None)
pipeline_str = (
    "udpsrc port=5600 caps=application/x-rtp,media=video,clock-rate=90000,encoding-name=H264 ! "
    "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
    "video/x-raw,format=BGR ! appsink name=sink emit-signals=true max-buffers=1 drop=true"
)
pipeline = Gst.parse_launch(pipeline_str)
appsink = pipeline.get_by_name("sink")
pipeline.set_state(Gst.State.PLAYING)

# =======================
# YOLO
# =======================
model = YOLO("/home/nizar/yolov5/runs/train4/exp_yolov8n/weights/best.pt")
MODEL_NAMES = model.names
print("[YOLO classes]", MODEL_NAMES)

# =======================
# MAVLINK
# =======================
print("[INFO] Araca bağlanılıyor...")
master = mavutil.mavlink_connection('udp:127.0.0.1:14551')
master.wait_heartbeat()
print(f"[INFO] Bağlandı: SYS={master.target_system} COMP={master.target_component}")

hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=2)
if hb is not None:
    try:
        sys_id  = hb.get_srcSystem()
        comp_id = hb.get_srcComponent()
        if sys_id:
            master.target_system = sys_id
        master.target_component = comp_id if comp_id not in (None, 0) else 1
    except Exception:
        if master.target_component in (None, 0):
            master.target_component = 1
print(f"[INFO] Target set: SYS={master.target_system} COMP={master.target_component}")

try:
    master.set_mode_apm('GUIDED')
except Exception:
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4
    )

master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0, 1, 0,0,0,0,0,0
)

t0 = time.time()
while time.time() - t0 < 8:
    hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
    if hb and (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
        print("[INFO] Armed.")
        break

# =======================
# MAV: KOMUT FONKSİYONLARI
# =======================
TYPE_MASK_POS_YAW = ((1<<3)|(1<<4)|(1<<5)|(1<<6)|(1<<7)|(1<<8)|(1<<11))

def goto_waypoint(lat, lon, alt, yaw_angle_deg=0.0):
    """
    GNSS tabanlı konum hedefi. Yaw’u da aynı pakette gönderebiliyoruz.
    Son WPT’de pozisyon tutmak için bu fonksiyon düzenli aralıklarla çağrılır.
    """
    yaw_rad = math.radians(yaw_angle_deg)
    master.mav.set_position_target_global_int_send(
        0, master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        TYPE_MASK_POS_YAW,
        int(lat*1e7), int(lon*1e7), float(alt),
        0,0,0, 0,0,0,
        yaw_rad, 0
    )

def set_yaw_target(yaw_angle_deg):
    """
    Sadece dönüş komutu gerektiğinde kullanılır (agresif kaçınma, bypass vs.)
    """
    yaw_radians = math.radians(yaw_angle_deg)
    msg = master.mav.set_position_target_local_ned_encode(
        0, master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b100111111111,  # sadece yaw aktif
        0,0,0, 0,0,0, 0,0,0,
        yaw_radians, 0
    )
    master.mav.send(msg)

def set_ground_speed(speed_mps: float):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
        0, 1, float(speed_mps), -1, 0, 0, 0, 0
    )

# =======================
# COĞRAFİ HESAPLAR
# =======================
R_EARTH = 6371000.0
def distance_2d_m(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2): return float("inf")
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1); dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2.0)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2.0)**2
    a = max(0.0, min(1.0, a)); c = 2.0 * math.asin(math.sqrt(a))
    return R_EARTH * c

def initial_bearing_deg(lat1, lon1, lat2, lon2):
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    λ1, λ2 = math.radians(lon1), math.radians(lon2)
    dλ = λ2 - λ1
    y = math.sin(dλ) * math.cos(φ2)
    x = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(dλ)
    θ = math.degrees(math.atan2(y, x))
    return (θ + 360.0) % 360.0

def dest_point_from_bearing(lat, lon, bearing_deg, dist_m):
    br = math.radians(bearing_deg)
    dN = dist_m * math.cos(br)
    dE = dist_m * math.sin(br)
    lat_new = lat + (dN / R_EARTH) * (180.0 / math.pi)
    lon_new = lon + (dE / (R_EARTH * math.cos(math.radians(lat)))) * (180.0 / math.pi)
    return lat_new, lon_new

# --- Yerel ENU approx (hat takibi için) ---
def ll_to_local_m(lat, lon, lat_ref, lon_ref):
    dN = (lat - lat_ref) * (math.pi/180.0) * R_EARTH
    dE = (lon - lon_ref) * (math.pi/180.0) * R_EARTH * math.cos(math.radians(lat_ref))
    return dE, dN  # x=East, y=North

def signed_cross_track_m(px, py, ax, ay, bx, by):
    vx, vy = (bx-ax), (by-ay)
    wx, wy = (px-ax), (py-ay)
    L = max(1e-6, math.hypot(vx, vy))
    return (vx*wy - vy*wx) / L  # sol(+), sağ(-)

def wrap180(a):
    while a > 180.0: a -= 360.0
    while a < -180.0: a += 360.0
    return a

def wrap360(a):
    while a >= 360.0: a -= 360.0
    while a < 0.0:    a += 360.0
    return a

# =======================
# DURUM OKUMA
# =======================
_last_lat = _last_lon = _last_alt = None
def get_current_position():
    """
    GLOBAL_POSITION_INT parse edip son geçerli (lat,lon,alt_rel) döndürür.
    Mesaj gelmezse son değeri korur.
    """
    global _last_lat, _last_lon, _last_alt
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
    if msg is not None:
        _last_lat = msg.lat / 1e7
        _last_lon = msg.lon / 1e7
        _last_alt = (msg.relative_alt/1000.0) if hasattr(msg, 'relative_alt') else _last_alt
    return _last_lat, _last_lon, _last_alt

# =======================
# RENK / FÜZYON
# =======================
def awb_grayworld(img):
    b,g,r = cv2.split(img)
    kb,kg,kr = b.mean(), g.mean(), r.mean()
    k = (kb+kg+kr)/3.0
    b = (b*(k/(kb+1e-6))).clip(0,255).astype(np.uint8)
    g = (g*(k/(kg+1e-6))).clip(0,255).astype(np.uint8)
    r = (r*(k/(kr+1e-6))).clip(0,255).astype(np.uint8)
    return cv2.merge([b,g,r])

_gamma_lut = None
def apply_gamma(img, gamma=1.1):
    global _gamma_lut
    if _gamma_lut is None or len(_gamma_lut)!=256:
        x=np.arange(256,dtype=np.float32)/255.0
        _gamma_lut=np.clip((x**(1.0/gamma))*255.0,0,255).astype(np.uint8)
    return cv2.LUT(img,_gamma_lut)

def classify_color_hsv_lab_with_scores(frame, x1, y1, x2, y2):
    x1=max(0,x1); y1=max(0,y1)
    x2=min(frame.shape[1]-1,x2); y2=min(frame.shape[0]-1,y2)
    w=x2-x1; h=y2-y1
    if w<8 or h<8: return "unknown", {}
    yb1=y1+int(0.40*h)
    roi=frame[yb1:y2, x1:x2]
    if roi.size==0: return "unknown", {}
    roi = apply_gamma(awb_grayworld(roi), gamma=1.1)

    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    lab=cv2.cvtColor(roi,cv2.COLOR_BGR2LAB)
    S=hsv[...,1]; V=hsv[...,2]
    mask=(S>=70)&(V>=70)
    if mask.sum()<80:
        if (V<50).sum()>0.15*V.size: return "black", {"black":1.0}
        return "unknown", {}

    H=hsv[...,0][mask]
    a=lab[...,1][mask].astype(np.float32)-128.0
    b=lab[...,2][mask].astype(np.float32)-128.0
    a_mean,b_mean=float(a.mean()), float(b.mean())

    def frac(lo,hi):
        sel=(H>=lo)&(H<=hi) if lo<=hi else ((H>=lo)|(H<=hi))
        return sel.sum()/float(H.size)

    red_r   = max(frac(0,9), frac(171,179))
    orange_r= frac(8,24)
    yellow_r= frac(25,38)
    green_r = frac(39,85)

    red_bonus    = max(0.0,(a_mean-10.0)-max(0.0,b_mean-6.0))*0.0045
    orange_bonus = max(0.0,(b_mean-12.0)-max(0.0,a_mean-6.0))*0.0045

    scores={"red":red_r+red_bonus, "orange":orange_r+orange_bonus,
            "yellow":yellow_r, "green":green_r}
    cname=max(scores, key=scores.get)
    if scores[cname]<0.12: return "unknown", scores
    return cname, scores

def fuse_model_and_color(raw_name, cls_conf, c_hsvlab, scores):
    rn = raw_name.lower() if raw_name else ""
    if ("orange" in rn or "yellow" in rn) and (cls_conf is None or cls_conf>=0.60):
        return "orange" if "orange" in rn else "yellow"
    if "red" in rn and (cls_conf is not None and cls_conf>=0.75):
        if scores.get("orange",0.0) > scores.get("red",0.0)+0.04: return "orange"
        return "red"
    return c_hsvlab

COLORS={"yellow":(0,255,255),"orange":(0,165,255),"green":(0,255,0),
        "red":(0,0,255),"black":(0,0,0),"unknown":(255,255,255)}
def draw_box(img,x1,y1,x2,y2,cname,conf):
    col = COLORS.get(cname,(255,255,255))
    if cname=="black": cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,255),3)
    cv2.rectangle(img,(x1,y1),(x2,y2),col,2)
    label=f"{cname} {conf:.2f}" if conf is not None else cname
    (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
    cv2.rectangle(img,(x1,y1-22),(x1+tw+6,y1),col,-1)
    cv2.putText(img,label,(x1+3,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.55,
                (0,0,0) if cname!="black" else (255,255,255),2)

def draw_ui_frame(vis):
    h,w = vis.shape[:2]
    cv2.rectangle(vis,(6,6),(w-6,h-6),(255,255,255),2)
    margin = int(w * (1.0 - CENTER_BAND) / 2.0)
    xL, xR = margin, w - margin
    yTop, yBot = int(0.20*h), int(0.95*h)
    cv2.rectangle(vis,(xL,yTop),(xR,yBot),(255,215,0),1)
    overlay=vis.copy()
    cv2.rectangle(overlay,(0,yTop),(xL,yBot),(0,0,0),-1)
    cv2.rectangle(overlay,(xR,yTop),(w,yBot),(0,0,0),-1)
    cv2.addWeighted(overlay,0.12,vis,0.88,0,vis)
    cv2.line(vis,(w//2,yTop),(w//2,yBot),(200,200,200),1,cv2.LINE_AA)
    cv2.putText(vis,"CENTER BAND",(xL+4,yTop-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,215,0),1,cv2.LINE_AA)

def estimate_range_from_bbox(x1,y1,x2,y2,img_w,img_h):
    w_px = max(1, x2 - x1); h_px = max(1, y2 - y1)
    fy = (img_h/2.0) / math.tan(math.radians(CAMERA_VFOV_DEG/2.0))
    fx = (img_w/2.0) / math.tan(math.radians(CAMERA_HFOV_DEG/2.0))
    r_h = (fy * BUOY_HEIGHT_M) / float(h_px)
    r_w = (fx * BUOY_WIDTH_M)  / float(w_px)
    r = min(r_h, r_w)
    if y2 >= img_h * NEAR_PIX_BOTTOM_FRAC: r = min(r, 4.0)
    area_ratio = (w_px * h_px) / float(img_w * img_h)
    if area_ratio >= AREA_NEAR_FRAC: r = min(r, 6.0)
    return max(0.5, min(60.0, r)), r_h, r_w

_prev_center_h = None; _prev_center_t = None; _ttc_ema = None
last_front_seen_time = 0.0
def reset_ttc():
    global _prev_center_h, _prev_center_t, _ttc_ema
    _prev_center_h = None; _prev_center_t = None; _ttc_ema = None
def update_ttc(h_now, t_now):
    global _prev_center_h, _prev_center_t, _ttc_ema
    ttc_inst = None
    if _prev_center_h is not None and _prev_center_t is not None:
        dt = max(1e-3, t_now - _prev_center_t); dh = (h_now - _prev_center_h)
        if dh > 1.0:
            ttc_inst = h_now / (dh / dt); ttc_inst = max(0.1, min(30.0, ttc_inst))
    if ttc_inst is not None:
        _ttc_ema = ttc_inst if _ttc_ema is None else TTC_EMA_ALPHA * _ttc_ema + (1.0 - TTC_EMA_ALPHA) * ttc_inst
    _prev_center_h = h_now; _prev_center_t = t_now
    return _ttc_ema

# =======================
# MAVLINK RENK KODU DİNLEME
# =======================
def parse_color_str(s: str):
    """STATUSTEXT 'COLOR: XXX' / 'RENK: XXX' içindeki stringleri eşle."""
    if not s: return None
    t = s.strip().lower()
    if t in ("kirmizi","kırmızı","red","r"):       return "red"
    if t in ("sari","sarı","yellow","y"):          return "yellow"
    if t in ("turuncu","orange","o"):              return "orange"
    if t in ("yesil","yeşil","green","g"):         return "green"
    if t in ("siyah","black","bk","b"):            return "black"
    return None

def parse_color_id(v: int):
    """Sayı → renk haritalaması (0..4)."""
    try:
        n = int(v)
    except Exception:
        return None
    mapping = {0:"red", 1:"yellow", 2:"green", 3:"orange", 4:"black"}
    return mapping.get(n, None)

last_color_msg_text = ""
def poll_color_code():
    """
    Non-blocking MAVLink okuma: Renk kodu (string) yakalarsa döndürür; yoksa None.
    Kabul edilenler:
      - STATUSTEXT: "COLOR: RED" / "RENK: YESIL" gibi
      - NAMED_VALUE_INT: name∈{"COLOR","RENK","COLOR_ID","RENK_ID"} ; value 0..4
      - PARAM_VALUE: param_id aynı; param_value 0..4
    DİKKAT: Bu fonksiyon bir mesaj tüketir; sadece colorwait modunda çağırıyoruz.
    """
    global last_color_msg_text
    msg = master.recv_match(blocking=False)
    if msg is None:
        return None

    mtype = msg.get_type()

    if mtype == "STATUSTEXT":
        txt = getattr(msg, "text", "") or ""
        up = txt.upper()
        if up.startswith("COLOR:") or up.startswith("RENK:"):
            payload = txt.split(":", 1)[1].strip()
            code = parse_color_str(payload)
            if code:
                last_color_msg_text = f"STATUSTEXT '{txt}'"
                return code

    elif mtype == "NAMED_VALUE_INT":
        name = (getattr(msg, "name", "") or "").upper()
        if name in ("COLOR","RENK","COLOR_ID","RENK_ID"):
            val = int(getattr(msg, "value", 0))
            code = parse_color_id(val)
            if code:
                last_color_msg_text = f"NAMED_VALUE_INT {name}={val}"
                return code

    elif mtype == "PARAM_VALUE":
        pid = getattr(msg, "param_id", "")
        if isinstance(pid, bytes):
            try: pid = pid.decode("ascii","ignore")
            except Exception: pid = ""
        if (pid or "").upper() in ("COLOR","RENK","COLOR_ID","RENK_ID"):
            val = int(getattr(msg, "param_value", 0))
            code = parse_color_id(val)
            if code:
                last_color_msg_text = f"PARAM_VALUE {pid}={val}"
                return code

    return None

# =======================
# KAÇINMA + TETİKLER
# =======================
def compute_avoid_and_vis(frame, results):
    vis = frame.copy()
    h, w = frame.shape[:2]
    margin = int(w * (1.0 - CENTER_BAND) / 2.0)
    xL, xR = margin, w - margin
    yTop, yBot = int(0.20*h), int(0.95*h)
    dead_half_base = int((DEAD_AHEAD_BAND * w) * 0.5)

    repulsion = total_area = danger_center = left_area = right_area = 0.0
    nearest_center_dist = None; nearest_center_bbox = None
    nearest_left_dist = None; nearest_right_dist = None

    r0 = results[0]; boxes = getattr(r0,"boxes",None)
    if boxes is None:
        draw_ui_frame(vis)
        cv2.putText(vis, "avoid_deg=0.0",(10,30), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)
        return 0.0, LOOKAHEAD_BASE_M, vis, (False, False, None, +1, None, False, False, 0, None, False)

    for b in boxes:
        conf = float(b.conf[0]) if b.conf is not None else None
        if conf is not None and conf < CONF_THR: continue

        if b.xyxy is not None: x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
        elif b.xywh is not None:
            cx0,cy0,bw,bh = b.xywh[0].tolist()
            x1=int(cx0-bw/2); y1=int(cy0-bh/2); x2=int(cx0+bw/2); y2=int(cy0+bh/2)
        else: continue

        raw = ""
        try:
            cls_id = int(b.cls[0]) if b.cls is not None else None
            raw = MODEL_NAMES.get(cls_id,"") if isinstance(MODEL_NAMES,dict) else ""
        except Exception: raw = ""
        c_hsvlab, scores = classify_color_hsv_lab_with_scores(frame, x1,y1,x2,y2)
        cname = fuse_model_and_color(raw, conf, c_hsvlab, scores)

        draw_box(vis,x1,y1,x2,y2,cname,conf if conf is not None else 1.0)

        est_r, r_h, r_w = estimate_range_from_bbox(x1,y1,x2,y2,w,h)
        cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
        inside_band = (xL <= cx <= xR) and (yTop <= cy <= yBot)

        dead_half = int(dead_half_base * 0.6)
        dead_xL, dead_xR = (w//2 - dead_half), (w//2 + dead_half)
        inside_dead = (dead_xL <= cx <= dead_xR)

        if inside_dead and inside_band and (cname in AVOID_COLOR_SET or cname=="unknown"):
            if (nearest_center_dist is None) or (est_r < nearest_center_dist):
                nearest_center_dist = est_r; nearest_center_bbox = (x1,y1,x2,y2)

        if not inside_band:
            cv2.putText(vis,"ignored",(x1+3,max(0,y1-26)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1)
            continue

        if   cname in AVOID_COLOR_SET: gain = COLOR_GAIN
        elif cname == "unknown":       gain = UNKNOWN_GAIN
        else:                          gain = NONTARGET_GAIN

        dx_norm = ((cx - (w*0.5)) / (w*0.5))
        area_ratio  = ((x2-x1)*(y2-y1)) / float(w*h)
        width_ratio = (x2-x1) / float(w)
        center_factor = 1.0 - min(1.0, abs(dx_norm))
        base = 40.0

        repulsion += gain * (area_ratio*3.0 + width_ratio*1.2) * (-np.sign(dx_norm)) * base * (1.2 + 0.8*center_factor)

        total_area += area_ratio
        if dx_norm < 0:
            left_area  += area_ratio
            if (nearest_left_dist is None) or (est_r < nearest_left_dist): nearest_left_dist = est_r
        else:
            right_area += area_ratio
            if (nearest_right_dist is None) or (est_r < nearest_right_dist): nearest_right_dist = est_r

        if abs(dx_norm) < (CENTER_BAND*0.5): danger_center = max(danger_center, area_ratio)

    avoid_deg = repulsion
    if total_area > 0.015: avoid_deg *= 1.25
    if danger_center > 0.010:
        prefer_right = (left_area > right_area)  # sol doluysa sağa kaç
        avoid_deg += (CENTER_BOOST_DEG if prefer_right else -CENTER_BOOST_DEG)

    if nearest_center_dist is not None and nearest_center_dist < 7.0:
        k = max(0.0, (7.0 - nearest_center_dist) / 7.0)
        avoid_deg *= (1.0 + 0.25 * k)

    avoid_deg = max(-MAX_AVOID_DEG, min(MAX_AVOID_DEG, avoid_deg))
    avoid_deg = float(MAX_AVOID_DEG * math.tanh(avoid_deg / MAX_AVOID_DEG))

    lookahead = LOOKAHEAD_BASE_M
    if danger_center > 0.012:          lookahead = max(LOOKAHEAD_MIN_M, LOOKAHEAD_BASE_M - 3.0)
    elif total_area > 0.03:            lookahead = max(LOOKAHEAD_MIN_M, LOOKAHEAD_BASE_M - 2.0)
    lookahead = min(LOOKAHEAD_MAX_M, lookahead)

    ctr = (int(w*0.5), int(h*0.9)); L = 120
    ang = math.radians(90 - avoid_deg)
    p2 = (int(ctr[0] + L*math.cos(ang)), int(ctr[1] - L*math.sin(ang)))
    cv2.arrowedLine(vis, ctr, p2, (255,255,0), 3, tipLength=0.2)
    cv2.putText(vis, f"avoid_deg={avoid_deg:.1f}  lookahead={lookahead:.1f}m",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    draw_ui_frame(vis)

    prefer_right = (left_area > right_area)
    sugg_dir = +1 if prefer_right else -1

    ttc_ema = None; frontal_present = False
    if nearest_center_bbox is not None:
        x1,y1,x2,y2 = nearest_center_bbox
        hpx = max(1, y2-y1)
        ttc_ema = update_ttc(hpx, time.time())
        frontal_present = True

    preempt_trigger = False; emerg_trigger = False
    if nearest_center_dist is not None:
        if (nearest_center_dist <= FRONT_TRIG_DIST_M) or (ttc_ema is not None and ttc_ema <= EMERG_TTC_S): emerg_trigger = True
        if (nearest_center_dist <= PREEMPT_TRIG_DIST_M) or (ttc_ema is not None and ttc_ema <= PREEMPT_TTC_S): preempt_trigger = True

    conflict_trigger = False; conflict_dir = 0; conflict_min_dist = None
    left_ok  = (left_area  > CONFLICT_AREA_THR)  and (nearest_left_dist  is not None)
    right_ok = (right_area > CONFLICT_AREA_THR)  and (nearest_right_dist is not None)
    if left_ok and right_ok:
        conflict_min_dist = min(nearest_left_dist, nearest_right_dist)
        if conflict_min_dist <= CONFLICT_NEAR_DIST_M:
            if (nearest_left_dist + CONFLICT_DIST_MARGIN_M) < nearest_right_dist: conflict_dir = +1
            elif (nearest_right_dist + CONFLICT_DIST_MARGIN_M) < nearest_left_dist: conflict_dir = -1
            else: conflict_dir = (+1 if left_area > right_area else -1)
            conflict_trigger = True

    panic_trigger = False
    if nearest_center_dist is not None:
        if (nearest_center_dist <= PANIC_TRIG_DIST_M) or (ttc_ema is not None and ttc_ema <= PANIC_TTC_S):
            panic_trigger = True
            lookahead = max(LOOKAHEAD_MIN_M, 2.0)

    if nearest_center_dist is not None:
        txt = f"front_dist~{nearest_center_dist:.1f}m"
        if ttc_ema is not None: txt += f"  TTC~{ttc_ema:.1f}s"
        cv2.putText(vis, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,215,255), 2)

    return avoid_deg, lookahead, vis, (
        preempt_trigger, emerg_trigger, nearest_center_dist, sugg_dir, ttc_ema, frontal_present,
        conflict_trigger, conflict_dir, conflict_min_dist, panic_trigger
    )

# =======================
# ANA DÖNGÜ
# =======================
APPROACH_THRESH_M = 3.0
SEND_PERIOD = 0.2

speed_mode = "cruise"     # cruise | avoid | preempt | emerg | conflict | panic | recovery | guard | bypass | colorwait
last_speed_cmd = 0.0

front_active_state   = False
preempt_state        = False
preempt_hold_start   = None
emerg_cool_until     = 0.0
front_dir_state      = +1
preempt_commit_until = 0.0
emerg_commit_until   = 0.0
last_default_side    = +1

conflict_state        = False
conflict_dir_state    = +1
conflict_commit_until = 0.0

panic_state           = False
panic_dir_state       = +1
panic_commit_until    = 0.0

recovery_state        = False
recovery_dir_state    = +1
recovery_start_time   = 0.0
recovery_start_lat    = None
recovery_start_lon    = None

evade_active_prev     = False
last_evade_dir        = +1

corner_lock_until     = 0.0

guard_state        = False
guard_dir_state    = +1
guard_commit_until = 0.0

# --- BYPASS (kama) state
bypass_state        = False
bypass_dir_state    = +1
bypass_commit_until = 0.0
bypass_start_lat    = None
bypass_start_lon    = None

# --- STUCK izleme
pos_hist = []        # (t, lat, lon)
steer_hist = []      # (t, sign)  sign = -1/0/+1

# --- COLOR WAIT state ---
color_wait_state     = False
color_wait_start     = 0.0
color_code_rx        = None  # "red"/"yellow"/"green"/"orange"/"black"

try:
    set_ground_speed(CRUISE_SPEED_MPS)
    last_speed_cmd = time.time()
    last_nav_send = 0.0

    while True:
        # ---- Görüntü çek ----
        sample = appsink.emit("try-pull-sample", 100000000)
        if sample is None:
            get_current_position(); time.sleep(0.01); continue

        buf = sample.get_buffer()
        caps = sample.get_caps()
        width = caps.get_structure(0).get_value('width')
        height = caps.get_structure(0).get_value('height')
        data = buf.extract_dup(0, buf.get_size())
        frame = np.frombuffer(data, np.uint8).reshape((height, width, 3)).copy()

        # ---- Algılama ----
        results = model(frame, verbose=False)
        avoid_deg, lookahead_m, vis, finfo = compute_avoid_and_vis(frame, results)
        (preempt_now, emerg_now, front_dist, sugg_dir, ttc_ema, frontal_present,
         conflict_now, conflict_dir, conflict_min_dist, panic_now) = finfo

        now = time.time()

        # TTC unutma
        if frontal_present: last_front_seen_time = now
        else:
            if (now - last_front_seen_time) > TTC_FORGET_SEC: reset_ttc()

        # --- durum güncelle (panic/emerg/preempt/conflict/guard) ---
        # PANIC
        if panic_state:
            if (front_dist is None) or (front_dist >= FRONT_CLEAR_DIST_M and (ttc_ema is None or ttc_ema > PANIC_TTC_S*1.4)):
                panic_state = False; panic_commit_until = 0.0
        else:
            if panic_now:
                panic_state = True
                panic_dir_state = (sugg_dir if sugg_dir in (-1,+1) else last_default_side)
                last_default_side = -last_default_side
                panic_commit_until = now + PANIC_COMMIT_S

        # EMERG
        if not panic_state:
            if front_active_state:
                if (front_dist is None) or (front_dist >= FRONT_CLEAR_DIST_M and (ttc_ema is None or ttc_ema > EMERG_TTC_S*1.2)):
                    front_active_state = False
                    emerg_cool_until = now + 1.0
                    emerg_commit_until = 0.0
            else:
                if emerg_now and (now >= emerg_cool_until):
                    front_active_state = True
                    chosen = sugg_dir if sugg_dir in (-1,+1) else last_default_side
                    front_dir_state = chosen; last_default_side = -last_default_side
                    emerg_commit_until = now + EMERG_COMMIT_S

        # PREEMPT
        if not (panic_state or front_active_state):
            if preempt_state:
                if (front_dist is None) or (front_dist >= PREEMPT_CLEAR_DIST_M and (ttc_ema is None or ttc_ema > PREEMPT_TTC_S*1.2)):
                    preempt_state = False; preempt_hold_start = None; preempt_commit_until = 0.0
            else:
                if preempt_now:
                    if preempt_hold_start is None: preempt_hold_start = now
                    if (now - preempt_hold_start) >= 0.0:
                        preempt_state = True
                        chosen = sugg_dir if sugg_dir in (-1,+1) else last_default_side
                        front_dir_state = chosen; last_default_side = -last_default_side
                        preempt_commit_until = now + PREEMPT_COMMIT_S
        else:
            preempt_state = False; preempt_commit_until = 0.0

        # CONFLICT
        if not (panic_state or front_active_state or preempt_state):
            if conflict_state:
                if (not conflict_now) and (now >= conflict_commit_until): conflict_state = False
            else:
                if conflict_now:
                    conflict_state = True
                    conflict_dir_state = conflict_dir if conflict_dir in (-1,+1) else (sugg_dir if sugg_dir in (-1,+1) else last_default_side)
                    last_default_side = -last_default_side
                    conflict_commit_until = now + CONFLICT_COMMIT_S
        else:
            conflict_state = False; conflict_commit_until = 0.0

        # kaçış yönünü hatırla
        if   panic_state:         last_evade_dir = +1 if panic_dir_state>0 else -1
        elif front_active_state:  last_evade_dir = +1 if front_dir_state>0 else -1
        elif preempt_state:       last_evade_dir = +1 if front_dir_state>0 else -1
        elif conflict_state:      last_evade_dir = +1 if conflict_dir_state>0 else -1

        # RECOVERY giriş/çıkış
        current_evade_active = (panic_state or front_active_state or preempt_state or conflict_state)
        if evade_active_prev and (not current_evade_active):
            recovery_state = True
            recovery_dir_state  = last_evade_dir
            recovery_start_time = now
            latc_tmp, lonc_tmp, _ = get_current_position()
            recovery_start_lat, recovery_start_lon = latc_tmp, lonc_tmp

        if recovery_state:
            front_clear_ok = ((front_dist is None or front_dist > RESUME_FRONT_CLEAR_M) and
                              (ttc_ema is None or ttc_ema > RESUME_TTC_CLEAR_S))
            latc_tmp, lonc_tmp, _ = get_current_position()
            travelled_ok = False
            if (recovery_start_lat is not None) and (recovery_start_lon is not None) and (latc_tmp is not None) and (lonc_tmp is not None):
                drec = distance_2d_m(recovery_start_lat, recovery_start_lon, latc_tmp, lonc_tmp)
                travelled_ok = (drec >= RECOVERY_DIST_RESUME_M)
            time_ok = (now - recovery_start_time) >= RECOVERY_TIME_MAX_S
            if (front_clear_ok and (travelled_ok or time_ok)) or current_evade_active:
                recovery_state = False

        if current_evade_active: recovery_state = False
        evade_active_prev = current_evade_active

        # =======================
        # NAV + STUCK/BYPASS
        # =======================
        latc, lonc, altc = get_current_position()
        if latc is None or lonc is None:
            cv2.imshow("UDP Stream with YOLO", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # STUCK: konum geçmişini güncelle
        pos_hist.append((now, latc, lonc))
        while pos_hist and (now - pos_hist[0][0] > STUCK_WINDOW_S): pos_hist.pop(0)
        moved_window = 0.0
        if len(pos_hist) >= 2:
            t0_hist, lat0h, lon0h = pos_hist[0]
            t1_hist, lat1h, lon1h = pos_hist[-1]
            moved_window = distance_2d_m(lat0h, lon0h, lat1h, lon1h)
        # flip-stuck (son STUCK_FLIP_WINDOW_S içinde sağ/sol flip sayısı)
        t_cut = now - STUCK_FLIP_WINDOW_S
        flips = 0; last_sign = None
        for (ts, sgn) in [x for x in steer_hist if x[0] >= t_cut]:
            if sgn == 0: continue
            if last_sign is None: last_sign = sgn
            elif sgn != last_sign:
                flips += 1; last_sign = sgn

        stuck_by_move = ( (pos_hist[-1][0] - pos_hist[0][0]) >= STUCK_WINDOW_S*0.6 and moved_window < STUCK_MOVE_THRESH_M )
        stuck_by_flip = (flips >= STUCK_FLIP_COUNT)

        if not (0 <= targetCoordinate < len(coordinates)):
            print("[NAV] Geçerli WP yok, çıkılıyor."); break

        lat_t, lon_t, alt_t = coordinates[targetCoordinate]
        dist = distance_2d_m(latc, lonc, lat_t, lon_t)

        # WPT değişimi
        if dist < APPROACH_THRESH_M:
            if targetCoordinate < len(coordinates) - 1:
                targetCoordinate += 1
                print(f"[NAV] Next WP -> idx={targetCoordinate} {coordinates[targetCoordinate]}")
                lat_t, lon_t, alt_t = coordinates[targetCoordinate]
                corner_lock_until = now + CORNER_LOCK_S
                # Yeni hedefe geçildi; mesafeyi yeni hedefe göre güncelle
                dist = distance_2d_m(latc, lonc, lat_t, lon_t)
            else:
                # Son WPT: burada hedef değiştirmiyoruz; colorwait devralacak
                print("[NAV] All waypoints reached; holding last WP.")

        # --- SON WPT'DE DUR/BEKLE (COLOR WAIT) ---
        if COLOR_WAIT_ENABLE:
            is_last_wp = (targetCoordinate == len(coordinates) - 1)
            dist_last = distance_2d_m(latc, lonc, coordinates[-1][0], coordinates[-1][1])

            if is_last_wp and (dist_last <= APPROACH_THRESH_M):
                if not color_wait_state:
                    color_wait_state = True
                    color_wait_start = now
                    # Hızı minimuma indir (fiilen duruş), konumu sabitle
                    set_ground_speed(STOP_SPEED_MPS)
                    print("[COLOR WAIT] Son WPT'de duruldu; renk kodu bekleniyor...")

                # Sadece bekleme modundayken MAVLink renk kodu dinle
                code_now = poll_color_code()
                if code_now:
                    color_code_rx = code_now
                    print(f"[COLOR WAIT] Renk kodu alındı: {color_code_rx}  Kaynak: {last_color_msg_text}")

                # İsteğe bağlı timeout (sadece log; hareket yok)
                if (COLOR_WAIT_TIMEOUT_S is not None) and ((now - color_wait_start) >= COLOR_WAIT_TIMEOUT_S):
                    print("[COLOR WAIT] Zaman aşımı; bekleme sürüyor (konum tutuluyor).")

                # Pozisyonu sık sık aynı WPT'ye kilitle (drift'e karşı)
                if (time.time() - last_nav_send) > SEND_PERIOD:
                    goto_waypoint(coordinates[-1][0], coordinates[-1][1], coordinates[-1][2], 0.0)
                    last_nav_send = time.time()

                # HUD/Overlay
                cv2.putText(vis, f"mode=colorwait  v_set={STOP_SPEED_MPS:.2f} m/s",
                            (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 2)
                y_info = 90
                cv2.putText(vis, "COLOR WAIT @ LAST WP", (10, y_info),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2); y_info += 24
                if color_code_rx:
                    cv2.putText(vis, f"RX COLOR = {color_code_rx.upper()}",
                                (10, y_info), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2); y_info += 22
                cv2.putText(vis, f"WP#{targetCoordinate} dist={dist_last:.1f} m (LAST)",
                            (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (50,220,50), 2)

                # Bekleme modunda NAV/kaçınma kararları uygulanmaz; sonraki döngüye geç
                cv2.imshow("UDP Stream with YOLO", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue  # <--- CRITICAL: beklerken kalan NAV mantığını atlıyoruz

        # --- Color-wait aktif değilse normal NAV devam eder ---
        bearing_to_wp  = initial_bearing_deg(latc, lonc, lat_t, lon_t)
        desired_course = bearing_to_wp

        # Hat takibi (sakin durumdayken)
        if LINE_TRACK_ENABLE and targetCoordinate > 0:
            lat_prev, lon_prev, _ = coordinates[targetCoordinate-1]
            ax, ay = ll_to_local_m(lat_prev, lon_prev, lat0, lon0)
            bx, by = ll_to_local_m(lat_t,    lon_t,    lat0, lon0)
            px, py = ll_to_local_m(latc,     lonc,     lat0, lon0)
            line_hdg = initial_bearing_deg(lat_prev, lon_prev, lat_t, lon_t)
            cte = signed_cross_track_m(px, py, ax, ay, bx, by)
            if (not (panic_state or front_active_state or preempt_state or conflict_state or recovery_state or guard_state or bypass_state)) and (abs(avoid_deg) <= LINE_USE_THRESH_DEG):
                stanley_deg = math.degrees(math.atan2(STANLEY_K * cte, max(0.5, 0.5)))
                stanley_deg = max(-STANLEY_MAX_DEG, min(STANLEY_MAX_DEG, stanley_deg))
                desired_course = wrap360(line_hdg + stanley_deg)
                if abs(stanley_deg) > 8.0:
                    lookahead_m = max(LOOKAHEAD_MIN_M, min(lookahead_m, 2.8))

        # Köşe kilidi
        bias_deg = 0.0
        if (now < corner_lock_until) and (not (panic_state or front_active_state or preempt_state or conflict_state or recovery_state or guard_state or bypass_state)) and (abs(avoid_deg) <= 8.0):
            if LINE_TRACK_ENABLE and targetCoordinate > 0:
                lat_prev, lon_prev, _ = coordinates[targetCoordinate-1]
                line_hdg = initial_bearing_deg(lat_prev, lon_prev, lat_t, lon_t)
                turn_err = wrap180(line_hdg - bearing_to_wp)
                turn_dir = 1 if turn_err > 0 else -1
            else:
                turn_dir = 1 if wrap180(desired_course - bearing_to_wp) > 0 else -1
            bias_deg += CORNER_EXTRA_BIAS * turn_dir
            lookahead_m = min(lookahead_m, CORNER_LOOKAHEAD_M)

        # Guard tetik/çıkış
        near_wp = (dist <= NEAR_WP_R_M)
        if near_wp and (not guard_state) and (not (panic_state or front_active_state or preempt_state or conflict_state or bypass_state)):
            risk_front = (panic_now or emerg_now or preempt_now or (front_dist is not None and front_dist <= FRONT_TRIG_DIST_M+1.0))
            if risk_front:
                guard_state = True
                guard_dir_state = (sugg_dir if sugg_dir in (-1,+1) else last_default_side)
                guard_commit_until = now + GUARD_COMMIT_S
        if guard_state:
            front_clear_ok = ((front_dist is None or front_dist > RESUME_FRONT_CLEAR_M) and (ttc_ema is None or ttc_ema > RESUME_TTC_CLEAR_S))
            if (now >= guard_commit_until and front_clear_ok) or (panic_state or front_active_state or preempt_state or conflict_state) or (not near_wp):
                guard_state = False

        # BYPASS tetik
        near_wp_bypass = (dist <= BYPASS_NEAR_WP_R_M)
        if (not bypass_state) and near_wp_bypass and (not (panic_state or front_active_state or preempt_state or conflict_state or recovery_state)):
            risk_front = (panic_now or emerg_now or preempt_now or (front_dist is not None and front_dist <= FRONT_TRIG_DIST_M+1.0))
            if risk_front and (stuck_by_move or stuck_by_flip):
                bypass_state = True
                bypass_dir_state = (sugg_dir if sugg_dir in (-1,+1) else -last_evade_dir)
                bypass_commit_until = now + BYPASS_COMMIT_S
                bypass_start_lat, bypass_start_lon, _ = get_current_position()

        # BYPASS çıkış
        if bypass_state:
            front_clear_ok = ((front_dist is None or front_dist > RESUME_FRONT_CLEAR_M) and
                              (ttc_ema is None or ttc_ema > RESUME_TTC_CLEAR_S))
            travel_ok = False
            lat_now, lon_now, _ = get_current_position()
            if (bypass_start_lat is not None) and (bypass_start_lon is not None) and (lat_now is not None) and (lon_now is not None):
                d_bypass = distance_2d_m(bypass_start_lat, bypass_start_lon, lat_now, lon_now)
                travel_ok = (d_bypass >= BYPASS_MIN_TRAVEL_M)
            time_ok = (now >= bypass_commit_until)
            if (time_ok and front_clear_ok) or travel_ok or (panic_state or front_active_state or preempt_state or conflict_state) or (not near_wp_bypass):
                bypass_state = False

        # Durum bias’ları
        if   panic_state:
            near_scale = 0.0
            if front_dist is not None: near_scale = max(0.0, min(1.0, (PANIC_TRIG_DIST_M - front_dist) / PANIC_TRIG_DIST_M))
            bias_p = PANIC_BIAS_DEG * (1 if panic_dir_state>0 else -1) + (12.0 * near_scale) * (1 if panic_dir_state>0 else -1)
            if abs(ALPHA_AVOID*avoid_deg + bias_p) < FRONTAL_PUSH_MIN_DEG:
                bias_p = FRONTAL_PUSH_MIN_DEG * (1 if panic_dir_state>0 else -1)
            bias_deg += bias_p

        elif front_active_state:
            near_scale = 0.0
            if front_dist is not None: near_scale = max(0.0, min(1.0, (FRONT_TRIG_DIST_M - front_dist) / FRONT_TRIG_DIST_M))
            bias_e = FRONT_BIAS_DEG * front_dir_state + (10.0 * near_scale) * front_dir_state
            if abs(ALPHA_AVOID*avoid_deg + bias_e) < FRONTAL_PUSH_MIN_DEG:
                bias_e = FRONTAL_PUSH_MIN_DEG * (1 if front_dir_state>0 else -1)
            bias_deg += bias_e

        elif preempt_state:
            bias_deg += PREEMPT_BIAS_DEG * front_dir_state

        elif conflict_state:
            extra = 0.0
            if conflict_min_dist is not None:
                extra = max(0.0, min(6.0, (CONFLICT_NEAR_DIST_M - conflict_min_dist))) * 0.8
            bias_deg += (CONFLICT_BIAS_DEG + extra) * (1 if conflict_dir_state>0 else -1)

        elif recovery_state:
            bias_deg += RECOVERY_BIAS_DEG * (1 if recovery_dir_state>0 else -1)
            lookahead_m = min(lookahead_m, RECOVERY_LOOKAHEAD_M)

        elif guard_state:
            lookahead_m = min(lookahead_m, GUARD_LOOKAHEAD_M)

        # Yakın-WP’de kaçınmayı biraz güçlendir
        alphaA = ALPHA_AVOID * (ALPHA_AVOID_NEAR_GAIN if near_wp else 1.0)

        # BYPASS aktifken: yan kurs
        if bypass_state:
            side_brg = wrap360(bearing_to_wp + (BYPASS_ANGLE_DEG if bypass_dir_state>0 else -BYPASS_ANGLE_DEG))
            desired_course = side_brg
            lookahead_m = min(lookahead_m, BYPASS_LOOKAHEAD_M)
            alphaA *= BYPASS_ALPHA_GAIN
            bias_deg += BYPASS_EXTRA_BIAS * (1 if bypass_dir_state>0 else -1)

        blended_bearing = wrap360(desired_course + alphaA * avoid_deg + bias_deg)
        lat_v, lon_v = dest_point_from_bearing(latc, lonc, blended_bearing, lookahead_m)

        # --- HIZ MODU ---
        avoid_mag = abs(avoid_deg)
        desired_mode = "cruise"
        if   panic_state:        desired_mode = "panic"
        elif front_active_state: desired_mode = "emerg"
        elif preempt_state:      desired_mode = "preempt"
        elif conflict_state:     desired_mode = "conflict"
        elif bypass_state:       desired_mode = "bypass"
        elif guard_state:        desired_mode = "guard"
        elif recovery_state:     desired_mode = "recovery"
        elif avoid_mag >= 12.0:  desired_mode = "avoid"

        target_speed = {
            "cruise":  CRUISE_SPEED_MPS,
            "avoid":   AVOID_SPEED_MPS,
            "preempt": PREEMPT_SPEED_MPS,
            "emerg":   EMERG_SPEED_MPS,
            "conflict":CONFLICT_SPEED_MPS,
            "panic":   PANIC_SPEED_MPS,
            "recovery":RECOVERY_SPEED_MPS,
            "guard":   GUARD_SPEED_MPS,
            "bypass":  BYPASS_SPEED_MPS,
            "colorwait": STOP_SPEED_MPS
        }[desired_mode]

        if (panic_state or front_active_state or preempt_state) and (ttc_ema is not None):
            if   ttc_ema < STOP_TTC_S: target_speed = min(target_speed, STOP_SPEED_MPS)
            elif ttc_ema < 2.0:        target_speed = min(target_speed, 0.12)
            elif ttc_ema < 3.0:        target_speed = min(target_speed, 0.18)
            elif ttc_ema < 4.0:        target_speed = min(target_speed, 0.24)

        if not (panic_state or front_active_state or preempt_state or conflict_state or recovery_state or guard_state or bypass_state):
            if desired_mode == "cruise": target_speed = max(target_speed, MIN_CRUISE_FLOOR_MPS)
            elif desired_mode == "avoid": target_speed = max(target_speed, MIN_AVOID_FLOOR_MPS)

        if (desired_mode != speed_mode) or (now - last_speed_cmd > SPEED_CMD_REFRESH_SEC):
            set_ground_speed(target_speed)
            speed_mode = desired_mode
            last_speed_cmd = now

        # --- KOMUT GÖNDER ---
        if (time.time() - last_nav_send) > SEND_PERIOD:
            steer_cmd = alphaA * avoid_deg + bias_deg
            # Agresif/özel modlarda yaw ile zorla döndür; diğerlerinde waypoint it
            if (panic_state or front_active_state or preempt_state or conflict_state or recovery_state or guard_state or bypass_state or (abs(steer_cmd) > 12.0)):
                set_yaw_target(steer_cmd)
            else:
                goto_waypoint(lat_v, lon_v, alt_t, 0)
            last_nav_send = time.time()

            # steer flip izleme
            sgn = 0
            if steer_cmd > 2.0: sgn = +1
            elif steer_cmd < -2.0: sgn = -1
            steer_hist.append((now, sgn))
            while steer_hist and (now - steer_hist[0][0] > STUCK_FLIP_WINDOW_S): steer_hist.pop(0)

        # HUD
        cv2.putText(vis, f"mode={speed_mode}  v_set={target_speed:.2f} m/s",
                    (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 2)
        yhud = 90
        if guard_state:
            cv2.putText(vis, f"GUARD -> {'RIGHT' if guard_dir_state>0 else 'LEFT'} (near WP)", (10, yhud), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,215,255), 2); yhud += 22
        if bypass_state:
            cv2.putText(vis, f"BYPASS -> {'RIGHT' if bypass_dir_state>0 else 'LEFT'} (stuck near WP)", (10, yhud), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2); yhud += 22
        if (stuck_by_move or stuck_by_flip):
            cv2.putText(vis, f"STUCK? move={moved_window:.1f}m flips={flips}", (10, yhud), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,140,255), 2)

        cv2.putText(vis, f"WP#{targetCoordinate} dist={dist:.1f} m  brg_toWP={bearing_to_wp:.1f}  course={desired_course:.1f}",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (50,220,50), 2)

        cv2.imshow("UDP Stream with YOLO", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Ctrl+C ile durduruldu.")
finally:
    pipeline.set_state(Gst.State.NULL)
    cv2.destroyAllWindows()
    print("[INFO] Pipeline kapatildi.")
