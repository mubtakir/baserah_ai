#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# animated_path_plotter_timeline.py
# Author: [باسل يحيى عبدالله] (مع تحسينات وإضافة دعم التحريك والتحكم الزمني)
# Version: 2.1
# Description: A Python class using Matplotlib to parse a custom drawing language
#              and render 2D vector graphics with animation support and timeline controls.

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.animation as animation
from matplotlib.colors import hex2color, to_hex
import matplotlib.widgets as widgets
import re
import numpy as np
import math
import time
import os
from enum import Enum, auto
import json
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

class AnimationEasing(Enum):
    """أنواع منحنيات التوقيت للتحريك"""
    LINEAR = auto()
    EASE_IN = auto()
    EASE_OUT = auto()
    EASE_IN_OUT = auto()
    BOUNCE = auto()
    ELASTIC = auto()
    BACK = auto()
    SINE = auto()

class AnimatedProperty(Enum):
    """الخصائص القابلة للتحريك"""
    POSITION = auto()
    SCALE = auto()
    ROTATION = auto()
    COLOR = auto()
    FILL_COLOR = auto()
    OPACITY = auto()
    THICKNESS = auto()
    FILL_ENABLED = auto()

class Keyframe:
    """
    فئة تمثل إطار رئيسي (keyframe) في التحريك
    """
    def __init__(self, time: float, value: Any, easing: AnimationEasing = AnimationEasing.LINEAR):
        """
        تهيئة إطار رئيسي جديد
        
        Args:
            time: الزمن بالثواني
            value: قيمة الخاصية في هذا الإطار
            easing: نوع منحنى التوقيت
        """
        self.time = time
        self.value = value
        self.easing = easing
    
    def __lt__(self, other):
        """للمقارنة والترتيب حسب الزمن"""
        return self.time < other.time

class AnimatedObject:
    """
    فئة تمثل كائن قابل للتحريك
    """
    def __init__(self, name: str):
        """
        تهيئة كائن متحرك جديد
        
        Args:
            name: اسم الكائن
        """
        self.name = name
        self.parent = None
        self.children = []
        self.visible = True
        
        # قاموس يحتوي على الإطارات الرئيسية لكل خاصية
        self.keyframes = {
            AnimatedProperty.POSITION: [],
            AnimatedProperty.SCALE: [],
            AnimatedProperty.ROTATION: [],
            AnimatedProperty.COLOR: [],
            AnimatedProperty.FILL_COLOR: [],
            AnimatedProperty.OPACITY: [],
            AnimatedProperty.THICKNESS: [],
            AnimatedProperty.FILL_ENABLED: []
        }
        
        # القيم الافتراضية للخصائص
        self.default_values = {
            AnimatedProperty.POSITION: (0, 0),
            AnimatedProperty.SCALE: 1.0,
            AnimatedProperty.ROTATION: 0.0,
            AnimatedProperty.COLOR: "#000000",
            AnimatedProperty.FILL_COLOR: "#808080",
            AnimatedProperty.OPACITY: 1.0,
            AnimatedProperty.THICKNESS: 1.0,
            AnimatedProperty.FILL_ENABLED: False
        }
        
        # أوامر الرسم الخاصة بهذا الكائن
        self.drawing_commands = []
    
    def add_keyframe(self, prop: AnimatedProperty, time: float, value: Any, 
                    easing: AnimationEasing = AnimationEasing.LINEAR):
        """
        إضافة إطار رئيسي لخاصية معينة
        
        Args:
            prop: الخاصية المراد تحريكها
            time: الزمن بالثواني
            value: قيمة الخاصية في هذا الإطار
            easing: نوع منحنى التوقيت
        """
        keyframe = Keyframe(time, value, easing)
        self.keyframes[prop].append(keyframe)
        # ترتيب الإطارات الرئيسية حسب الزمن
        self.keyframes[prop].sort()
    
    def get_value_at_time(self, prop: AnimatedProperty, time: float) -> Any:
        """
        الحصول على قيمة خاصية في زمن معين
        
        Args:
            prop: الخاصية المطلوبة
            time: الزمن بالثواني
            
        Returns:
            قيمة الخاصية في الزمن المحدد
        """
        keyframes = self.keyframes[prop]
        
        # إذا لم تكن هناك إطارات رئيسية، استخدم القيمة الافتراضية
        if not keyframes:
            return self.default_values[prop]
        
        # إذا كان الزمن قبل أول إطار رئيسي، استخدم قيمة أول إطار
        if time <= keyframes[0].time:
            return keyframes[0].value
        
        # إذا كان الزمن بعد آخر إطار رئيسي، استخدم قيمة آخر إطار
        if time >= keyframes[-1].time:
            return keyframes[-1].value
        
        # البحث عن الإطارين الرئيسيين المحيطين بالزمن المطلوب
        for i in range(len(keyframes) - 1):
            if keyframes[i].time <= time < keyframes[i+1].time:
                # حساب نسبة الزمن بين الإطارين
                t_range = keyframes[i+1].time - keyframes[i].time
                t_progress = (time - keyframes[i].time) / t_range
                
                # تطبيق منحنى التوقيت
                t_eased = self._apply_easing(t_progress, keyframes[i+1].easing)
                
                # حساب القيمة المتوسطة حسب نوع الخاصية
                return self._interpolate_value(
                    prop, keyframes[i].value, keyframes[i+1].value, t_eased
                )
    
    def _apply_easing(self, t: float, easing: AnimationEasing) -> float:
        """
        تطبيق منحنى التوقيت على نسبة الزمن
        
        Args:
            t: نسبة الزمن (0.0 - 1.0)
            easing: نوع منحنى التوقيت
            
        Returns:
            نسبة الزمن بعد تطبيق منحنى التوقيت
        """
        if easing == AnimationEasing.LINEAR:
            return t
        elif easing == AnimationEasing.EASE_IN:
            return t * t
        elif easing == AnimationEasing.EASE_OUT:
            return t * (2 - t)
        elif easing == AnimationEasing.EASE_IN_OUT:
            return t * t * (3 - 2 * t)
        elif easing == AnimationEasing.BOUNCE:
            # منحنى ارتداد
            if t < 0.5:
                return 4 * t * t
            else:
                return 4 * (t - 1) * (t - 1) + 1
        elif easing == AnimationEasing.ELASTIC:
            # منحنى مرن
            return math.sin(13 * math.pi/2 * t) * math.pow(2, 10 * (t - 1))
        elif easing == AnimationEasing.BACK:
            # منحنى رجوع
            s = 1.70158
            return t * t * ((s + 1) * t - s)
        elif easing == AnimationEasing.SINE:
            # منحنى جيبي
            return math.sin(t * math.pi/2)
        else:
            return t
    
    def _interpolate_value(self, prop: AnimatedProperty, start_val: Any, 
                          end_val: Any, t: float) -> Any:
        """
        حساب القيمة المتوسطة بين قيمتين حسب نوع الخاصية
        
        Args:
            prop: نوع الخاصية
            start_val: القيمة البدائية
            end_val: القيمة النهائية
            t: نسبة الزمن (0.0 - 1.0)
            
        Returns:
            القيمة المتوسطة
        """
        if prop == AnimatedProperty.POSITION:
            # تحريك الموقع (x, y)
            start_x, start_y = start_val
            end_x, end_y = end_val
            return (
                start_x + (end_x - start_x) * t,
                start_y + (end_y - start_y) * t
            )
        elif prop == AnimatedProperty.SCALE:
            # تحريك الحجم
            return start_val + (end_val - start_val) * t
        elif prop == AnimatedProperty.ROTATION:
            # تحريك الدوران
            return start_val + (end_val - start_val) * t
        elif prop in [AnimatedProperty.COLOR, AnimatedProperty.FILL_COLOR]:
            # تحريك اللون
            start_rgb = hex2color(start_val)
            end_rgb = hex2color(end_val)
            
            r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * t
            g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * t
            b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * t
            
            return to_hex((r, g, b))
        elif prop == AnimatedProperty.OPACITY:
            # تحريك الشفافية
            return start_val + (end_val - start_val) * t
        elif prop == AnimatedProperty.THICKNESS:
            # تحريك سماكة الخط
            return start_val + (end_val - start_val) * t
        elif prop == AnimatedProperty.FILL_ENABLED:
            # تحريك حالة التعبئة (يتم التبديل عند t = 0.5)
            return end_val if t >= 0.5 else start_val
        else:
            return end_val

class TimelineController:
    """
    فئة للتحكم في الخط الزمني للتحريك
    """
    def __init__(self, fig, animation_manager):
        """
        تهيئة وحدة التحكم في الخط الزمني
        
        Args:
            fig: كائن Figure من matplotlib
            animation_manager: مدير التحريك
        """
        self.fig = fig
        self.animation_manager = animation_manager
        self.is_playing = False
        self.is_looping = False
        self.playback_speed = 1.0
        
        # إنشاء منطقة التحكم في الخط الزمني
        self.timeline_ax = plt.axes([0.1, 0.05, 0.8, 0.03])
        self.time_slider = widgets.Slider(
            self.timeline_ax, 'الزمن', 0, self.animation_manager.duration,
            valinit=0, valstep=1/self.animation_manager.fps
        )
        
        # إنشاء أزرار التحكم
        self.play_ax = plt.axes([0.3, 0.01, 0.1, 0.03])
        self.play_button = widgets.Button(self.play_ax, 'تشغيل')
        
        self.pause_ax = plt.axes([0.41, 0.01, 0.1, 0.03])
        self.pause_button = widgets.Button(self.pause_ax, 'إيقاف')
        
        self.stop_ax = plt.axes([0.52, 0.01, 0.1, 0.03])
        self.stop_button = widgets.Button(self.stop_ax, 'إعادة')
        
        self.loop_ax = plt.axes([0.63, 0.01, 0.1, 0.03])
        self.loop_button = widgets.Button(self.loop_ax, 'تكرار: لا')
        
        # ربط الأحداث بالأزرار
        self.play_button.on_clicked(self.play)
        self.pause_button.on_clicked(self.pause)
        self.stop_button.on_clicked(self.stop)
        self.loop_button.on_clicked(self.toggle_loop)
        self.time_slider.on_changed(self.on_time_changed)
        
        # إنشاء شريط سرعة التشغيل
        self.speed_ax = plt.axes([0.1, 0.01, 0.1, 0.03])
        self.speed_slider = widgets.Slider(
            self.speed_ax, 'السرعة', 0.1, 3.0,
            valinit=1.0, valstep=0.1
        )
        self.speed_slider.on_changed(self.on_speed_changed)
        
        # مؤقت التحديث
        self.timer = None
        self.last_update_time = None
    
    def play(self, event=None):
        """
        تشغيل التحريك
        
        Args:
            event: حدث النقر (اختياري)
        """
        if not self.is_playing:
            self.is_playing = True
            self.last_update_time = time.time()
            
            # إنشاء مؤقت لتحديث الخط الزمني
            self.timer = self.fig.canvas.new_timer(interval=1000/self.animation_manager.fps)
            self.timer.add_callback(self.update_time)
            self.timer.start()
    
    def pause(self, event=None):
        """
        إيقاف التحريك مؤقتاً
        
        Args:
            event: حدث النقر (اختياري)
        """
        if self.is_playing:
            self.is_playing = False
            if self.timer:
                self.timer.stop()
    
    def stop(self, event=None):
        """
        إيقاف التحريك وإعادته إلى البداية
        
        Args:
            event: حدث النقر (اختياري)
        """
        self.pause()
        self.time_slider.set_val(0)
        self.animation_manager.current_time = 0
    
    def toggle_loop(self, event=None):
        """
        تبديل حالة التكرار
        
        Args:
            event: حدث النقر (اختياري)
        """
        self.is_looping = not self.is_looping
        self.loop_button.label.set_text(f"تكرار: {'نعم' if self.is_looping else 'لا'}")
        self.fig.canvas.draw_idle()
    
    def on_time_changed(self, val):
        """
        معالجة تغيير الزمن من شريط التمرير
        
        Args:
            val: القيمة الجديدة للزمن
        """
        self.animation_manager.current_time = val
        self.fig.canvas.draw_idle()
    
    def on_speed_changed(self, val):
        """
        معالجة تغيير سرعة التشغيل
        
        Args:
            val: القيمة الجديدة للسرعة
        """
        self.playback_speed = val
    
    def update_time(self):
        """تحديث الزمن الحالي للتحريك"""
        if not self.is_playing:
            return
        
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # حساب الزمن الجديد
        new_time = self.animation_manager.current_time + elapsed * self.playback_speed
        
        # التعامل مع نهاية التحريك
        if new_time >= self.animation_manager.duration:
            if self.is_looping:
                # إعادة التشغيل من البداية
                new_time = new_time % self.animation_manager.duration
            else:
                # إيقاف التشغيل عند النهاية
                new_time = self.animation_manager.duration
                self.pause()
        
        # تحديث الزمن
        self.animation_manager.current_time = new_time
        self.time_slider.set_val(new_time)
        
        return True

class AnimationManager:
    """
    فئة لإدارة التحريك والتحكم في الكائنات المتحركة
    """
    def __init__(self):
        """تهيئة مدير التحريك"""
        self.objects = {}  # قاموس الكائنات المتحركة
        self.groups = {}   # قاموس مجموعات الكائنات
        self.paths = {}    # قاموس المسارات المعرفة
        
        # إعدادات التحريك
        self.duration = 10.0  # المدة الافتراضية بالثواني
        self.fps = 30         # معدل الإطارات الافتراضي
        self.output_format = "gif"  # تنسيق الإخراج الافتراضي
        self.save_filename = None   # اسم ملف الحفظ
        
        # متغيرات التشغيل
        self.current_time = 0.0
        self.is_playing = False
        self.animation_obj = None  # كائن التحريك من matplotlib
        
        # الكائن الحالي قيد التعريف
        self.current_object = None
        
        # قائمة الأحداث الزمنية
        self.time_events = []
        
        # وحدة التحكم في الخط الزمني
        self.timeline_controller = None
    
    def define_object(self, name: str):
        """
        تعريف كائن متحرك جديد
        
        Args:
            name: اسم الكائن
        """
        if name in self.objects:
            print(f"تحذير: الكائن '{name}' معرف مسبقاً. سيتم استبداله.")
        
        self.objects[name] = AnimatedObject(name)
        return self.objects[name]
    
    def begin_object_definition(self, name: str):
        """
        بدء تعريف كائن متحرك
        
        Args:
            name: اسم الكائن
        """
        if name not in self.objects:
            self.objects[name] = AnimatedObject(name)
        
        self.current_object = self.objects[name]
    
    def end_object_definition(self):
        """إنهاء تعريف الكائن الحالي"""
        self.current_object = None
    
    def define_group(self, name: str):
        """
        تعريف مجموعة كائنات جديدة
        
        Args:
            name: اسم المجموعة
        """
        if name in self.groups:
            print(f"تحذير: المجموعة '{name}' معرفة مسبقاً. سيتم استبدالها.")
        
        self.groups[name] = []
    
    def add_to_group(self, object_name: str, group_name: str):
        """
        إضافة كائن إلى مجموعة
        
        Args:
            object_name: اسم الكائن
            group_name: اسم المجموعة
        """
        if object_name not in self.objects:
            print(f"خطأ: الكائن '{object_name}' غير معرف.")
            return
        
        if group_name not in self.groups:
            self.groups[group_name] = []
        
        if object_name not in self.groups[group_name]:
            self.groups[group_name].append(object_name)
    
    def set_parent(self, child_name: str, parent_name: str):
        """
        تعيين علاقة أبوية بين كائنين
        
        Args:
            child_name: اسم الكائن الابن
            parent_name: اسم الكائن الأب
        """
        if child_name not in self.objects:
            print(f"خطأ: الكائن '{child_name}' غير معرف.")
            return
        
        if parent_name not in self.objects:
            print(f"خطأ: الكائن '{parent_name}' غير معرف.")
            return
        
        child = self.objects[child_name]
        parent = self.objects[parent_name]
        
        # إزالة الكائن من الأب السابق إن وجد
        if child.parent:
            old_parent = self.objects[child.parent]
            if child_name in old_parent.children:
                old_parent.children.remove(child_name)
        
        # تعيين الأب الجديد
        child.parent = parent_name
        if child_name not in parent.children:
            parent.children.append(child_name)
    
    def add_keyframe(self, object_name: str, prop: AnimatedProperty, 
                    time: float, value: Any, easing: AnimationEasing = AnimationEasing.LINEAR):
        """
        إضافة إطار رئيسي لكائن
        
        Args:
            object_name: اسم الكائن
            prop: الخاصية المراد تحريكها
            time: الزمن بالثواني
            value: قيمة الخاصية في هذا الإطار
            easing: نوع منحنى التوقيت
        """
        if object_name not in self.objects:
            print(f"خطأ: الكائن '{object_name}' غير معرف.")
            return
        
        obj = self.objects[object_name]
        obj.add_keyframe(prop, time, value, easing)
    
    def set_animation_duration(self, duration: float):
        """
        تعيين مدة التحريك
        
        Args:
            duration: المدة بالثواني
        """
        if duration <= 0:
            print("خطأ: مدة التحريك يجب أن تكون أكبر من صفر.")
            return
        
        self.duration = duration
    
    def set_animation_fps(self, fps: int):
        """
        تعيين معدل الإطارات
        
        Args:
            fps: عدد الإطارات في الثانية
        """
        if fps <= 0:
            print("خطأ: معدل الإطارات يجب أن يكون أكبر من صفر.")
            return
        
        self.fps = fps
    
    def set_animation_output(self, output_format: str):
        """
        تعيين تنسيق الإخراج
        
        Args:
            output_format: تنسيق الإخراج ("gif" أو "mp4")
        """
        if output_format.lower() not in ["gif", "mp4"]:
            print("خطأ: تنسيق الإخراج يجب أن يكون 'gif' أو 'mp4'.")
            return
        
        self.output_format = output_format.lower()
    
    def add_time_event(self, time: float, event_code: str):
        """
        إضافة حدث زمني
        
        Args:
            time: الزمن بالثواني
            event_code: كود الحدث
        """
        self.time_events.append((time, event_code))
        # ترتيب الأحداث حسب الزمن
        self.time_events.sort(key=lambda x: x[0])
    
    def setup_timeline_controller(self, fig):
        """
        إعداد وحدة التحكم في الخط الزمني
        
        Args:
            fig: كائن Figure من matplotlib
        """
        self.timeline_controller = TimelineController(fig, self)
    
    def update_frame(self, ax, time_override=None):
        """
        تحديث إطار التحريك
        
        Args:
            ax: كائن Axes من matplotlib
            time_override: زمن محدد للتحديث (اختياري)
            
        Returns:
            قائمة الأشكال المرسومة
        """
        # استخدام الزمن المحدد أو الزمن الحالي
        current_time = time_override if time_override is not None else self.current_time
        
        # مسح الرسم السابق
        ax.clear()
        
        # قائمة الأشكال المرسومة
        patches = []
        
        # رسم جميع الكائنات في الزمن الحالي
        for obj_name, obj in self.objects.items():
            if not obj.visible:
                continue
            
            # الحصول على قيم الخصائص في الزمن الحالي
            position = obj.get_value_at_time(AnimatedProperty.POSITION, current_time)
            scale = obj.get_value_at_time(AnimatedProperty.SCALE, current_time)
            rotation = obj.get_value_at_time(AnimatedProperty.ROTATION, current_time)
            color = obj.get_value_at_time(AnimatedProperty.COLOR, current_time)
            fill_color = obj.get_value_at_time(AnimatedProperty.FILL_COLOR, current_time)
            opacity = obj.get_value_at_time(AnimatedProperty.OPACITY, current_time)
            thickness = obj.get_value_at_time(AnimatedProperty.THICKNESS, current_time)
            fill_enabled = obj.get_value_at_time(AnimatedProperty.FILL_ENABLED, current_time)
            
            # رسم الكائن باستخدام الخصائص المحسوبة
            obj_patches = self._draw_object(
                ax, obj, position, scale, rotation, color, fill_color, 
                opacity, thickness, fill_enabled
            )
            patches.extend(obj_patches)
        
        # تنفيذ الأحداث الزمنية
        for event_time, event_code in self.time_events:
            if event_time <= current_time < event_time + 1/self.fps:
                # تنفيذ الحدث
                try:
                    exec(event_code)
                except Exception as e:
                    print(f"خطأ في تنفيذ الحدث: {e}")
        
        # إعداد الرسم
        ax.set_aspect('equal', adjustable='box')
        
        # ضبط حدود الرسم
        all_verts = []
        for patch in patches:
            path_obj = patch.get_path()
            if path_obj and len(path_obj.vertices) > 0:
                all_verts.extend(path_obj.vertices)
        
        if all_verts:
            all_verts_np = np.array(all_verts)
            if all_verts_np.size > 0:
                min_x, min_y = all_verts_np.min(axis=0)
                max_x, max_y = all_verts_np.max(axis=0)
                
                delta_x = max_x - min_x
                delta_y = max_y - min_y
                padding_x = delta_x * 0.1 if delta_x > 1e-6 else 1.0
                padding_y = delta_y * 0.1 if delta_y > 1e-6 else 1.0
                
                if np.isclose(delta_x, 0): padding_x = 1.0
                if np.isclose(delta_y, 0): padding_y = 1.0
                
                ax.set_xlim(min_x - padding_x, max_x + padding_x)
                ax.set_ylim(min_y - padding_y, max_y + padding_y)
            else:
                ax.set_xlim(-10, 10)
                ax.set_ylim(-10, 10)
        else:
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
        
        return patches
    
    def _draw_object(self, ax, obj, position, scale, rotation, color, fill_color, 
                    opacity, thickness, fill_enabled):
        """
        رسم كائن متحرك
        
        Args:
            ax: كائن Axes من matplotlib
            obj: الكائن المتحرك
            position: الموقع (x, y)
            scale: الحجم
            rotation: الدوران بالدرجات
            color: لون الخط
            fill_color: لون التعبئة
            opacity: الشفافية
            thickness: سماكة الخط
            fill_enabled: تفعيل التعبئة
            
        Returns:
            قائمة الأشكال المرسومة
        """
        patches = []
        
        # تطبيق التحويلات على أوامر الرسم
        for cmd_name, cmd_args in obj.drawing_commands:
            # تطبيق التحويلات على الأوامر حسب نوعها
            if cmd_name == "CIRCLE":
                if len(cmd_args) == 3:
                    # تحويل إحداثيات المركز ونصف القطر
                    cx, cy, radius = map(float, cmd_args)
                    
                    # تطبيق الموقع والحجم
                    cx = position[0] + cx * scale
                    cy = position[1] + cy * scale
                    radius = radius * scale
                    
                    # إنشاء الدائرة
                    circle = plt.Circle(
                        (cx, cy), radius,
                        fill=fill_enabled,
                        edgecolor=color,
                        facecolor=fill_color if fill_enabled else 'none',
                        linewidth=thickness,
                        alpha=opacity
                    )
                    
                    # تطبيق الدوران
                    if rotation != 0:
                        # الدوران حول المركز
                        transform = plt.matplotlib.transforms.Affine2D().rotate_deg_around(
                            cx, cy, rotation
                        ) + ax.transData
                        circle.set_transform(transform)
                    
                    # إضافة الدائرة إلى الرسم
                    ax.add_patch(circle)
                    patches.append(circle)
            
            elif cmd_name == "RECTANGLE":
                if len(cmd_args) == 4:
                    # تحويل إحداثيات المستطيل
                    x, y, width, height = map(float, cmd_args)
                    
                    # تطبيق الموقع والحجم
                    x = position[0] + x * scale
                    y = position[1] + y * scale
                    width = width * scale
                    height = height * scale
                    
                    # إنشاء المستطيل
                    rect = plt.Rectangle(
                        (x, y), width, height,
                        fill=fill_enabled,
                        edgecolor=color,
                        facecolor=fill_color if fill_enabled else 'none',
                        linewidth=thickness,
                        alpha=opacity
                    )
                    
                    # تطبيق الدوران
                    if rotation != 0:
                        # الدوران حول مركز المستطيل
                        center_x = x + width / 2
                        center_y = y + height / 2
                        transform = plt.matplotlib.transforms.Affine2D().rotate_deg_around(
                            center_x, center_y, rotation
                        ) + ax.transData
                        rect.set_transform(transform)
                    
                    # إضافة المستطيل إلى الرسم
                    ax.add_patch(rect)
                    patches.append(rect)
            
            # يمكن إضافة المزيد من أنواع الأشكال هنا
        
        return patches
    
    def create_animation(self, fig, ax, show_controls=True):
        """
        إنشاء التحريك باستخدام matplotlib
        
        Args:
            fig: كائن Figure من matplotlib
            ax: كائن Axes من matplotlib
            show_controls: إظهار عناصر التحكم في الخط الزمني
            
        Returns:
            كائن Animation من matplotlib
        """
        # إعداد وحدة التحكم في الخط الزمني
        if show_controls:
            self.setup_timeline_controller(fig)
        
        # حساب عدد الإطارات الكلي
        num_frames = int(self.duration * self.fps)
        
        # دالة تحديث الإطار
        def update(frame):
            # حساب الزمن الحالي
            time_val = frame / self.fps
            
            # تحديث الإطار
            patches = self.update_frame(ax, time_val)
            
            return patches
        
        # إنشاء كائن التحريك
        self.animation_obj = animation.FuncAnimation(
            fig, update, frames=num_frames, interval=1000/self.fps, blit=True
        )
        
        return self.animation_obj
    
    def save_animation(self, filename=None):
        """
        حفظ التحريك كملف
        
        Args:
            filename: اسم الملف (اختياري)
        """
        if not self.animation_obj:
            print("خطأ: يجب إنشاء التحريك أولاً باستخدام create_animation().")
            return
        
        # استخدام اسم الملف المحدد أو الاسم المخزن
        save_filename = filename if filename else self.save_filename
        
        if not save_filename:
            print("خطأ: لم يتم تحديد اسم ملف للحفظ.")
            return
        
        # التأكد من امتداد الملف
        if not save_filename.lower().endswith(f".{self.output_format}"):
            save_filename = f"{save_filename}.{self.output_format}"
        
        # حفظ التحريك
        try:
            if self.output_format == "gif":
                self.animation_obj.save(
                    save_filename, writer='pillow', fps=self.fps, 
                    dpi=100, savefig_kwargs={'facecolor': 'white'}
                )
            elif self.output_format == "mp4":
                self.animation_obj.save(
                    save_filename, writer='ffmpeg', fps=self.fps, 
                    dpi=100, savefig_kwargs={'facecolor': 'white'}
                )
            
            print(f"تم حفظ التحريك بنجاح في الملف: {save_filename}")
        except Exception as e:
            print(f"خطأ في حفظ التحريك: {e}")

class AnimatedPathPlotter:
    """
    فئة لرسم الأشكال المتحركة باستخدام لغة أوامر مخصصة
    """
    def __init__(self):
        """تهيئة الراسم المتحرك"""
        self.fig = None  # كائن Figure من matplotlib
        self.ax = None   # كائن Axes من matplotlib
        
        # إنشاء مدير التحريك
        self.animation_manager = AnimationManager()
        
        # حالة الرسم الحالية
        self.current_pen_down = False
        self.current_color_stroke = "#000000"
        self.current_thickness = 1.0
        self.current_opacity = 1.0
        self.current_fill_enabled = False
        self.current_color_fill = "#808080"
        self.current_path_vertices = []
        self.current_path_codes = []
        self.current_path_start_point = None
        self.last_point = (0, 0)
        self.patches_to_draw = []
        self.current_background_color = "white"
        self.save_filename = None
        self.save_dpi = 300
        self.num_variable_segments = 20
        
        # متغيرات خاصة بالتحريك
        self.current_object_name = None
        self.current_path_name = None
        self.is_recording_object = False
        self.is_recording_path = False
        self.recorded_commands = []
        
        # متغيرات التحكم في التشغيل
        self.show_timeline_controls = True
        self.auto_play = False
        self.loop_animation = False
    
    def _initialize_plot_area(self, figsize_override=None):
        """
        تهيئة منطقة الرسم
        
        Args:
            figsize_override: حجم الرسم (العرض، الارتفاع) بالإنش
        """
        if self.fig:
            try:
                plt.close(self.fig)
            except Exception:
                pass
        
        # تعديل حجم الرسم ليناسب عناصر التحكم
        if self.show_timeline_controls:
            figsize = figsize_override if figsize_override else (8, 7)  # زيادة الارتفاع لعناصر التحكم
        else:
            figsize = figsize_override if figsize_override else (8, 6)
        
        self.fig, self.ax = plt.subplots(figsize=figsize)
        
        # تعديل موضع الرسم ليناسب عناصر التحكم
        if self.show_timeline_controls:
            self.fig.subplots_adjust(bottom=0.2)  # ترك مساحة أسفل الرسم لعناصر التحكم
        
        self.ax.set_facecolor(self.current_background_color)
    
    def _reset_drawing_state(self):
        """إعادة تعيين حالة الرسم إلى القيم الافتراضية"""
        self.current_pen_down = False
        self.current_color_stroke = "#000000"
        self.current_thickness = 1.0
        self.current_opacity = 1.0
        self.current_fill_enabled = False
        self.current_color_fill = "#808080"
        self.current_path_vertices = []
        self.current_path_codes = []
        self.current_path_start_point = None
        self.last_point = (0, 0)
        self.patches_to_draw = []
        self.current_background_color = "white"
        self.save_filename = None
        self.save_dpi = 300
    
    def _clean_color_arg(self, color_arg: str) -> str:
        """
        تنظيف وسيط اللون من علامات الاقتباس
        
        Args:
            color_arg: وسيط اللون
            
        Returns:
            وسيط اللون بعد التنظيف
        """
        color_arg = color_arg.strip()
        if (color_arg.startswith('"') and color_arg.endswith('"')) or \
           (color_arg.startswith("'") and color_arg.endswith("'")):
            return color_arg[1:-1]
        return color_arg
    
    def _add_current_path_to_patches(self, is_closing_path=False):
        """
        إضافة المسار الحالي إلى قائمة الأشكال المراد رسمها
        
        Args:
            is_closing_path: هل المسار مغلق
        """
        if not self.current_path_codes:
            if is_closing_path:
                self.current_path_start_point = None
            return
        
        if len(self.current_path_vertices) != len(self.current_path_codes):
            print(f"خطأ: عدم تطابق بين عدد النقاط ({len(self.current_path_vertices)}) "
                  f"وعدد الأكواد ({len(self.current_path_codes)}).")
            self.current_path_vertices, self.current_path_codes = [], []
            if is_closing_path:
                self.current_path_start_point = None
            return
        
        path = Path(np.array(self.current_path_vertices), self.current_path_codes)
        
        facecolor_to_use = 'none'
        if self.current_fill_enabled:
            facecolor_to_use = self.current_color_fill
        
        edgecolor_to_use = 'none'
        if self.current_pen_down or \
           (is_closing_path and self.current_color_stroke != 'none' and self.current_thickness > 0):
            edgecolor_to_use = self.current_color_stroke
        
        should_create_patch = False
        if facecolor_to_use != 'none':
            should_create_patch = True
        if edgecolor_to_use != 'none' and self.current_thickness > 0:
            should_create_patch = True
        
        if not should_create_patch:
            self.current_path_vertices, self.current_path_codes = [], []
            if is_closing_path:
                 self.current_path_start_point = None
            return
        
        patch = PathPatch(
            path,
            facecolor=facecolor_to_use,
            edgecolor=edgecolor_to_use,
            lw=self.current_thickness if edgecolor_to_use != 'none' else 0,
            alpha=self.current_opacity,
            capstyle='round',
            joinstyle='round',
            fill=self.current_fill_enabled
        )
        self.patches_to_draw.append(patch)
        
        self.current_path_vertices = []
        self.current_path_codes = []
        
        if is_closing_path:
            self.current_path_start_point = None
    
    def _add_rectangle_path(self, x: float, y: float, width: float, height: float):
        """
        إضافة مسار مستطيل
        
        Args:
            x: الإحداثي السيني للزاوية السفلية اليسرى
            y: الإحداثي الصادي للزاوية السفلية اليسرى
            width: العرض
            height: الارتفاع
        """
        if self.current_path_codes:
            self._add_current_path_to_patches(is_closing_path=False)
        
        self.current_path_vertices = [
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
            [x, y]
        ]
        self.current_path_codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY
        ]
        self.current_path_start_point = (x,y)
        self._add_current_path_to_patches(is_closing_path=True)
        self.last_point = (x, y)
    
    def _add_circle_path(self, cx: float, cy: float, radius: float):
        """
        إضافة مسار دائرة
        
        Args:
            cx: الإحداثي السيني للمركز
            cy: الإحداثي الصادي للمركز
            radius: نصف القطر
        """
        if self.current_path_codes:
            self._add_current_path_to_patches(is_closing_path=False)
        
        kappa = 0.552284749831
        
        p = [
            (cx + radius, cy),
            (cx + radius, cy + kappa * radius),
            (cx + kappa * radius, cy + radius),
            (cx, cy + radius),
            
            (cx - kappa * radius, cy + radius),
            (cx - radius, cy + kappa * radius),
            (cx - radius, cy),
            
            (cx - radius, cy - kappa * radius),
            (cx - kappa * radius, cy - radius),
            (cx, cy - radius),
            
            (cx + kappa * radius, cy - radius),
            (cx + radius, cy - kappa * radius),
            (cx + radius, cy)
        ]
        
        self.current_path_vertices = [list(p[0])]
        self.current_path_codes = [Path.MOVETO]
        self.current_path_start_point = p[0]
        
        for i in range(0, 12, 3):
            self.current_path_vertices.extend([list(p[i+1]), list(p[i+2]), list(p[i+3])])
            self.current_path_codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
        
        self.current_path_codes[-1] = Path.CLOSEPOLY
        
        self._add_current_path_to_patches(is_closing_path=True)
        self.last_point = p[0]
    
    def _add_variable_thickness_line(self, p0: tuple, p1: tuple, thick0: float, thick1: float):
        """
        إضافة خط متغير السماكة
        
        Args:
            p0: نقطة البداية (x0, y0)
            p1: نقطة النهاية (x1, y1)
            thick0: السماكة عند نقطة البداية
            thick1: السماكة عند نقطة النهاية
        """
        if not self.current_pen_down:
            self.last_point = p1
            self.current_thickness = thick1
            return
        
        x0, y0 = p0
        x1, y1 = p1
        
        for i in range(self.num_variable_segments):
            t_seg_start = i / self.num_variable_segments
            t_seg_end = (i + 1) / self.num_variable_segments
            
            seg_x_start = x0 + (x1 - x0) * t_seg_start
            seg_y_start = y0 + (y1 - y0) * t_seg_start
            seg_x_end = x0 + (x1 - x0) * t_seg_end
            seg_y_end = y0 + (y1 - y0) * t_seg_end
            
            current_segment_thickness = thick0 + (thick1 - thick0) * (t_seg_start + t_seg_end) / 2.0
            
            if current_segment_thickness <= 1e-6:
                continue
            
            segment_verts = np.array([[seg_x_start, seg_y_start], [seg_x_end, seg_y_end]])
            segment_codes = [Path.MOVETO, Path.LINETO]
            
            segment_path = Path(segment_verts, segment_codes)
            segment_patch = PathPatch(
                segment_path,
                facecolor='none',
                edgecolor=self.current_color_stroke,
                lw=current_segment_thickness,
                alpha=self.current_opacity,
                capstyle='butt',
                joinstyle='round'
            )
            self.patches_to_draw.append(segment_patch)
        
        self.last_point = p1
        self.current_thickness = thick1
    
    def parse_and_execute(self, equation_string: str, figsize_override=None):
        """
        تحليل وتنفيذ أوامر الرسم
        
        Args:
            equation_string: سلسلة الأوامر
            figsize_override: حجم الرسم (العرض، الارتفاع) بالإنش
        """
        if not self.fig or not self.ax:
            self._initialize_plot_area(figsize_override)
        elif figsize_override and self.fig and not np.array_equal(self.fig.get_size_inches(), figsize_override):
            plt.close(self.fig)
            self._initialize_plot_area(figsize_override)
        
        current_bg_before_reset = self.current_background_color
        self._reset_drawing_state()
        self.current_background_color = current_bg_before_reset
        if self.ax:
            self.ax.set_facecolor(self.current_background_color)
        
        lines = equation_string.strip().split('\n')
        for line_num, line_content in enumerate(lines):
            line = line_content.strip()
            if not line or line.startswith('#'):
                continue
            
            command_name = ""
            args_str = ""
            match = re.match(r"([A-Z_]+)\s*\((.*)\)", line)
            if match:
                command_name = match.group(1).upper()
                args_str = match.group(2).strip()
            else:
                command_name = line.upper()
            
            raw_args = [self._clean_color_arg(a.strip()) for a in args_str.split(',') if a.strip()] if args_str else []
            
            # تسجيل الأمر إذا كنا في وضع التسجيل
            if self.is_recording_object:
                self.recorded_commands.append((command_name, raw_args))
            
            try:
                # أوامر الرسم الأساسية
                if command_name == "PEN_UP":
                    if self.current_pen_down and self.current_path_codes:
                        self._add_current_path_to_patches(is_closing_path=False)
                    self.current_pen_down = False
                elif command_name == "PEN_DOWN":
                    self.current_pen_down = True
                elif command_name == "SET_COLOR_HEX":
                    if raw_args:
                        self.current_color_stroke = raw_args[0]
                elif command_name == "SET_THICKNESS":
                    if raw_args:
                        self.current_thickness = float(raw_args[0])
                elif command_name == "SET_OPACITY":
                    if raw_args:
                        self.current_opacity = np.clip(float(raw_args[0]), 0.0, 1.0)
                elif command_name == "SET_FILL_COLOR_HEX":
                    if raw_args:
                        self.current_color_fill = raw_args[0]
                elif command_name == "ENABLE_FILL":
                    self.current_fill_enabled = True
                elif command_name == "DISABLE_FILL":
                    self.current_fill_enabled = False
                elif command_name == "MOVE_TO":
                    if len(raw_args) == 2:
                        if self.current_path_codes:
                            self._add_current_path_to_patches(is_closing_path=False)
                        x, y = float(raw_args[0]), float(raw_args[1])
                        self.current_path_vertices = [[x,y]]
                        self.current_path_codes = [Path.MOVETO]
                        self.current_path_start_point = (x,y)
                        self.last_point = (x,y)
                elif command_name == "LINE_TO":
                    if len(raw_args) == 2:
                        if not self.current_path_codes:
                            self.current_path_vertices.append(list(self.last_point))
                            self.current_path_codes.append(Path.MOVETO)
                            if self.current_path_start_point is None:
                                self.current_path_start_point = self.last_point
                        x, y = float(raw_args[0]), float(raw_args[1])
                        self.current_path_vertices.append([x,y])
                        self.current_path_codes.append(Path.LINETO)
                        self.last_point = (x,y)
                elif command_name == "CURVE_TO":
                    if len(raw_args) == 6:
                        if not self.current_path_codes:
                            self.current_path_vertices.append(list(self.last_point))
                            self.current_path_codes.append(Path.MOVETO)
                            if self.current_path_start_point is None:
                                self.current_path_start_point = self.last_point
                        c1x, c1y, c2x, c2y, ex, ey = map(float, raw_args)
                        self.current_path_vertices.extend([[c1x,c1y],[c2x,c2y],[ex,ey]])
                        self.current_path_codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
                        self.last_point = (ex,ey)
                elif command_name == "CLOSE_PATH":
                    if self.current_path_codes and self.current_path_start_point is not None:
                        self.current_path_vertices.append(list(self.current_path_start_point))
                        self.current_path_codes.append(Path.CLOSEPOLY)
                        self._add_current_path_to_patches(is_closing_path=True)
                        if self.current_path_start_point:
                            self.last_point = self.current_path_start_point
                elif command_name == "RECTANGLE":
                    if len(raw_args) == 4:
                        x, y, width, height = map(float, raw_args)
                        self._add_rectangle_path(x, y, width, height)
                elif command_name == "CIRCLE":
                    if len(raw_args) == 3:
                        cx, cy, radius = map(float, raw_args)
                        self._add_circle_path(cx, cy, radius)
                elif command_name == "SET_BACKGROUND_COLOR":
                    if raw_args:
                        self.current_background_color = raw_args[0]
                        if self.ax:
                            self.ax.set_facecolor(self.current_background_color)
                elif command_name == "SAVE_FIGURE":
                    if raw_args:
                        self.save_filename = raw_args[0]
                        if len(raw_args) > 1 and raw_args[1].startswith("dpi="):
                            try:
                                self.save_dpi = int(raw_args[1][4:])
                            except ValueError:
                                print(f"خطأ في السطر {line_num+1}: قيمة DPI غير صالحة.")
                elif command_name == "SET_VARIABLE_LINE_SEGMENTS":
                    if len(raw_args) == 1:
                        try:
                            segments = int(raw_args[0])
                            if segments > 0:
                                self.num_variable_segments = segments
                            else:
                                print(f"خطأ في السطر {line_num+1}: عدد القطع يجب أن يكون أكبر من صفر.")
                        except ValueError:
                            print(f"خطأ في السطر {line_num+1}: قيمة عدد القطع غير صالحة.")
                elif command_name == "VARIABLE_LINE_TO":
                    if len(raw_args) == 3:
                        if self.current_path_codes:
                            self._add_current_path_to_patches(is_closing_path=False)
                        
                        end_x = float(raw_args[0])
                        end_y = float(raw_args[1])
                        end_thickness = float(raw_args[2])
                        
                        start_point_var_line = self.last_point
                        start_thickness_var_line = self.current_thickness
                        
                        self._add_variable_thickness_line(
                            start_point_var_line,
                            (end_x, end_y),
                            start_thickness_var_line,
                            end_thickness
                        )
                
                # أوامر التحريك
                elif command_name == "SET_ANIMATION_DURATION":
                    if len(raw_args) == 1:
                        try:
                            duration = float(raw_args[0])
                            self.animation_manager.set_animation_duration(duration)
                        except ValueError:
                            print(f"خطأ في السطر {line_num+1}: قيمة المدة غير صالحة.")
                elif command_name == "SET_ANIMATION_FPS":
                    if len(raw_args) == 1:
                        try:
                            fps = int(raw_args[0])
                            self.animation_manager.set_animation_fps(fps)
                        except ValueError:
                            print(f"خطأ في السطر {line_num+1}: قيمة FPS غير صالحة.")
                elif command_name == "SET_ANIMATION_OUTPUT":
                    if len(raw_args) == 1:
                        self.animation_manager.set_animation_output(raw_args[0])
                elif command_name == "DEFINE_OBJECT":
                    if len(raw_args) == 1:
                        self.animation_manager.define_object(raw_args[0])
                elif command_name == "BEGIN_OBJECT":
                    if len(raw_args) == 1:
                        self.current_object_name = raw_args[0]
                        self.animation_manager.begin_object_definition(self.current_object_name)
                        self.is_recording_object = True
                        self.recorded_commands = []
                elif command_name == "END_OBJECT":
                    if self.is_recording_object and self.current_object_name:
                        obj = self.animation_manager.objects[self.current_object_name]
                        obj.drawing_commands = self.recorded_commands
                        self.animation_manager.end_object_definition()
                        self.is_recording_object = False
                        self.current_object_name = None
                elif command_name == "DEFINE_GROUP":
                    if len(raw_args) == 1:
                        self.animation_manager.define_group(raw_args[0])
                elif command_name == "ADD_TO_GROUP":
                    if len(raw_args) == 2:
                        self.animation_manager.add_to_group(raw_args[0], raw_args[1])
                elif command_name == "SET_PARENT":
                    if len(raw_args) == 2:
                        self.animation_manager.set_parent(raw_args[0], raw_args[1])
                elif command_name == "KEYFRAME_POSITION":
                    if len(raw_args) == 4:
                        obj_name = raw_args[0]
                        time = float(raw_args[1])
                        x = float(raw_args[2])
                        y = float(raw_args[3])
                        self.animation_manager.add_keyframe(
                            obj_name, AnimatedProperty.POSITION, time, (x, y)
                        )
                elif command_name == "KEYFRAME_SCALE":
                    if len(raw_args) == 3:
                        obj_name = raw_args[0]
                        time = float(raw_args[1])
                        scale = float(raw_args[2])
                        self.animation_manager.add_keyframe(
                            obj_name, AnimatedProperty.SCALE, time, scale
                        )
                elif command_name == "KEYFRAME_ROTATION":
                    if len(raw_args) == 3:
                        obj_name = raw_args[0]
                        time = float(raw_args[1])
                        rotation = float(raw_args[2])
                        self.animation_manager.add_keyframe(
                            obj_name, AnimatedProperty.ROTATION, time, rotation
                        )
                elif command_name == "KEYFRAME_COLOR":
                    if len(raw_args) == 3:
                        obj_name = raw_args[0]
                        time = float(raw_args[1])
                        color = raw_args[2]
                        self.animation_manager.add_keyframe(
                            obj_name, AnimatedProperty.COLOR, time, color
                        )
                elif command_name == "KEYFRAME_FILL_COLOR":
                    if len(raw_args) == 3:
                        obj_name = raw_args[0]
                        time = float(raw_args[1])
                        color = raw_args[2]
                        self.animation_manager.add_keyframe(
                            obj_name, AnimatedProperty.FILL_COLOR, time, color
                        )
                elif command_name == "KEYFRAME_OPACITY":
                    if len(raw_args) == 3:
                        obj_name = raw_args[0]
                        time = float(raw_args[1])
                        opacity = float(raw_args[2])
                        self.animation_manager.add_keyframe(
                            obj_name, AnimatedProperty.OPACITY, time, opacity
                        )
                elif command_name == "KEYFRAME_THICKNESS":
                    if len(raw_args) == 3:
                        obj_name = raw_args[0]
                        time = float(raw_args[1])
                        thickness = float(raw_args[2])
                        self.animation_manager.add_keyframe(
                            obj_name, AnimatedProperty.THICKNESS, time, thickness
                        )
                elif command_name == "KEYFRAME_FILL_ENABLED":
                    if len(raw_args) == 3:
                        obj_name = raw_args[0]
                        time = float(raw_args[1])
                        fill_enabled = raw_args[2].lower() in ["true", "1", "yes", "y"]
                        self.animation_manager.add_keyframe(
                            obj_name, AnimatedProperty.FILL_ENABLED, time, fill_enabled
                        )
                elif command_name == "SET_EASING":
                    if len(raw_args) == 3:
                        obj_name = raw_args[0]
                        prop_name = raw_args[1].upper()
                        easing_name = raw_args[2].upper()
                        
                        # تحويل اسم الخاصية إلى نوع AnimatedProperty
                        prop = None
                        if prop_name == "POSITION":
                            prop = AnimatedProperty.POSITION
                        elif prop_name == "SCALE":
                            prop = AnimatedProperty.SCALE
                        elif prop_name == "ROTATION":
                            prop = AnimatedProperty.ROTATION
                        elif prop_name == "COLOR":
                            prop = AnimatedProperty.COLOR
                        elif prop_name == "FILL_COLOR":
                            prop = AnimatedProperty.FILL_COLOR
                        elif prop_name == "OPACITY":
                            prop = AnimatedProperty.OPACITY
                        elif prop_name == "THICKNESS":
                            prop = AnimatedProperty.THICKNESS
                        elif prop_name == "FILL_ENABLED":
                            prop = AnimatedProperty.FILL_ENABLED
                        else:
                            print(f"خطأ في السطر {line_num+1}: اسم الخاصية '{prop_name}' غير صالح.")
                            continue
                        
                        # تحويل اسم منحنى التوقيت إلى نوع AnimationEasing
                        easing = None
                        if easing_name == "LINEAR":
                            easing = AnimationEasing.LINEAR
                        elif easing_name == "EASE_IN":
                            easing = AnimationEasing.EASE_IN
                        elif easing_name == "EASE_OUT":
                            easing = AnimationEasing.EASE_OUT
                        elif easing_name == "EASE_IN_OUT":
                            easing = AnimationEasing.EASE_IN_OUT
                        elif easing_name == "BOUNCE":
                            easing = AnimationEasing.BOUNCE
                        elif easing_name == "ELASTIC":
                            easing = AnimationEasing.ELASTIC
                        elif easing_name == "BACK":
                            easing = AnimationEasing.BACK
                        elif easing_name == "SINE":
                            easing = AnimationEasing.SINE
                        else:
                            print(f"خطأ في السطر {line_num+1}: اسم منحنى التوقيت '{easing_name}' غير صالح.")
                            continue
                        
                        # تطبيق منحنى التوقيت على جميع الإطارات الرئيسية للخاصية
                        if obj_name in self.animation_manager.objects:
                            obj = self.animation_manager.objects[obj_name]
                            for keyframe in obj.keyframes[prop]:
                                keyframe.easing = easing
                        else:
                            print(f"خطأ في السطر {line_num+1}: الكائن '{obj_name}' غير معرف.")
                elif command_name == "AT_TIME":
                    if len(raw_args) >= 2:
                        time = float(raw_args[0])
                        event_code = args_str[args_str.find(',') + 1:].strip()
                        self.animation_manager.add_time_event(time, event_code)
                elif command_name == "SHOW_TIMELINE_CONTROLS":
                    if len(raw_args) == 1:
                        self.show_timeline_controls = raw_args[0].lower() in ["true", "1", "yes", "y"]
                elif command_name == "AUTO_PLAY":
                    if len(raw_args) == 1:
                        self.auto_play = raw_args[0].lower() in ["true", "1", "yes", "y"]
                elif command_name == "LOOP_ANIMATION":
                    if len(raw_args) == 1:
                        self.loop_animation = raw_args[0].lower() in ["true", "1", "yes", "y"]
                elif command_name == "PLAY_ANIMATION":
                    # سيتم تنفيذ هذا الأمر لاحقاً في دالة plot
                    pass
                elif command_name == "SAVE_ANIMATION":
                    if len(raw_args) == 1:
                        # سيتم تنفيذ هذا الأمر لاحقاً في دالة plot
                        self.animation_manager.save_filename = raw_args[0]
                else:
                    print(f"خطأ في السطر {line_num+1}: الأمر '{command_name}' غير معروف.")
            except ValueError as ve:
                print(f"خطأ في السطر {line_num+1}: قيمة غير صالحة للأمر {command_name} - {ve}")
            except Exception as e:
                print(f"خطأ في السطر {line_num+1}: خطأ غير متوقع في معالجة الأمر {command_name} - {e}")
                import traceback
                traceback.print_exc()
        
        # إضافة أي مسار مفتوح متبقي
        if self.current_path_codes:
            self._add_current_path_to_patches(is_closing_path=False)
    
    def plot(self, title="Path Plotter Output", show_grid=True, figsize_override=None, animate=False):
        """
        رسم الأشكال
        
        Args:
            title: عنوان الرسم
            show_grid: إظهار الشبكة
            figsize_override: حجم الرسم (العرض، الارتفاع) بالإنش
            animate: تفعيل التحريك
        """
        if not self.ax:
            self._initialize_plot_area(figsize_override)
            print("تحذير: لم يتم تهيئة الراسم بالبيانات. سيتم عرض رسم فارغ.")
        elif figsize_override:
            current_fig_size = self.fig.get_size_inches()
            if not np.array_equal(current_fig_size, figsize_override):
                 self.fig.set_size_inches(figsize_override[0], figsize_override[1])
        
        if self.ax:
            self.ax.clear()
            self.ax.set_facecolor(self.current_background_color)
        
        if not self.patches_to_draw and self.ax and not animate:
            print("معلومات: لا توجد أشكال للرسم. سيتم عرض رسم فارغ.")
        
        if self.ax:
            if animate:
                # إنشاء التحريك
                animation_obj = self.animation_manager.create_animation(
                    self.fig, self.ax, show_controls=self.show_timeline_controls
                )
                
                # تعيين حالة التكرار
                if self.animation_manager.timeline_controller:
                    self.animation_manager.timeline_controller.is_looping = self.loop_animation
                    self.animation_manager.timeline_controller.loop_button.label.set_text(
                        f"تكرار: {'نعم' if self.loop_animation else 'لا'}"
                    )
                
                # تشغيل التحريك تلقائياً
                if self.auto_play and self.animation_manager.timeline_controller:
                    self.animation_manager.timeline_controller.play()
                
                # حفظ التحريك إذا تم تحديد اسم ملف
                if self.animation_manager.save_filename:
                    self.animation_manager.save_animation(self.animation_manager.save_filename)
            else:
                # رسم الأشكال الثابتة
                for patch in self.patches_to_draw:
                    self.ax.add_patch(patch)
            
            self.ax.set_aspect('equal', adjustable='box')
            
            # ضبط حدود الرسم
            all_verts_for_scaling = []
            if self.patches_to_draw:
                for p_item in self.patches_to_draw:
                    path_obj = p_item.get_path()
                    if path_obj and len(path_obj.vertices) > 0:
                        all_verts_for_scaling.extend(path_obj.vertices)
            
            if all_verts_for_scaling:
                all_verts_np = np.array(all_verts_for_scaling)
                if all_verts_np.size > 0:
                    min_x, min_y = all_verts_np.min(axis=0)
                    max_x, max_y = all_verts_np.max(axis=0)
                    
                    delta_x = max_x - min_x
                    delta_y = max_y - min_y
                    padding_x = delta_x * 0.1 if delta_x > 1e-6 else 1.0
                    padding_y = delta_y * 0.1 if delta_y > 1e-6 else 1.0
                    
                    if np.isclose(delta_x, 0): padding_x = 1.0
                    if np.isclose(delta_y, 0): padding_y = 1.0
                    
                    self.ax.set_xlim(min_x - padding_x, max_x + padding_x)
                    self.ax.set_ylim(min_y - padding_y, max_y + padding_y)
                else:
                    self.ax.set_xlim(-10, 10)
                    self.ax.set_ylim(-10, 10)
            else:
                self.ax.set_xlim(-10, 10)
                self.ax.set_ylim(-10, 10)
            
            self.ax.set_title(title)
            self.ax.set_xlabel("X-axis")
            self.ax.set_ylabel("Y-axis")
            if show_grid:
                self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # حفظ الرسم إذا تم تحديد اسم ملف
        if self.fig and self.save_filename and not animate:
            try:
                self.fig.savefig(self.save_filename, dpi=self.save_dpi, bbox_inches='tight')
                print(f"تم حفظ الرسم بنجاح في الملف: {self.save_filename} (DPI: {self.save_dpi}).")
            except Exception as e:
                print(f"خطأ في حفظ الرسم: {e}")
        
        # عرض الرسم
        if self.fig:
            plt.show()
        else:
            print("خطأ: كائن Figure غير متاح للرسم.")

# مثال على الاستخدام
if __name__ == "__main__":
    plotter = AnimatedPathPlotter()
    
    # مثال على أوامر التحريك مع عناصر التحكم
    animation_commands = """
    # إعداد التحريك
    SET_ANIMATION_DURATION(5)
    SET_ANIMATION_FPS(30)
    SET_ANIMATION_OUTPUT("bouncing_ball.gif")
    
    # إعدادات التحكم
    SHOW_TIMELINE_CONTROLS(true)
    AUTO_PLAY(true)
    LOOP_ANIMATION(true)
    
    # تعريف كائن الكرة
    DEFINE_OBJECT("ball")
    BEGIN_OBJECT("ball")
        PEN_DOWN()
        SET_COLOR_HEX("#FF0000")
        ENABLE_FILL()
        SET_FILL_COLOR_HEX("#FF0000")
        CIRCLE(0, 0, 20)
        PEN_UP()
    END_OBJECT()
    
    # تحريك الكرة
    KEYFRAME_POSITION("ball", 0, 50, 50)
    KEYFRAME_POSITION("ball", 1, 150, 150)
    KEYFRAME_POSITION("ball", 2, 250, 50)
    KEYFRAME_POSITION("ball", 3, 350, 150)
    KEYFRAME_POSITION("ball", 4, 450, 50)
    
    # تغيير حجم الكرة
    KEYFRAME_SCALE("ball", 0, 1.0)
    KEYFRAME_SCALE("ball", 2, 2.0)
    KEYFRAME_SCALE("ball", 4, 1.0)
    
    # تغيير لون الكرة
    KEYFRAME_COLOR("ball", 0, "#FF0000")
    KEYFRAME_COLOR("ball", 2, "#00FF00")
    KEYFRAME_COLOR("ball", 4, "#0000FF")
    
    # تعيين منحنيات التوقيت
    SET_EASING("ball", "position", "ease_in_out")
    SET_EASING("ball", "scale", "bounce")
    SET_EASING("ball", "color", "linear")
    
    # حفظ التحريك
    SAVE_ANIMATION("bouncing_ball.gif")
    """
    
    plotter.parse_and_execute(animation_commands)
    plotter.plot(title="Animated Path Plotter Demo with Timeline Controls", animate=True)
