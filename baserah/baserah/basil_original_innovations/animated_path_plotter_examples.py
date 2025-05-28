#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# animated_path_plotter_examples.py
# Author: [باسل يحيى عبدالله] (مع تحسينات وإضافة دعم التحريك)
# Version: 2.2
# Description: أمثلة متنوعة على استخدام برنامج رسم الأشكال المتحركة

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from animated_path_plotter_timeline import AnimatedPathPlotter

def example_bouncing_ball():
    """مثال على كرة متحركة مع تغيير اللون والحجم"""
    plotter = AnimatedPathPlotter()
    
    commands = """
    # إعداد التحريك
    SET_ANIMATION_DURATION(5)
    SET_ANIMATION_FPS(30)
    SET_ANIMATION_OUTPUT("bouncing_ball.gif")
    SET_BACKGROUND_COLOR(#F0F0F0)
    
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
    
    # تحريك الكرة (مسار قفز)
    KEYFRAME_POSITION("ball", 0, 50, 200)
    KEYFRAME_POSITION("ball", 0.5, 150, 50)
    KEYFRAME_POSITION("ball", 1.0, 250, 200)
    KEYFRAME_POSITION("ball", 1.5, 350, 50)
    KEYFRAME_POSITION("ball", 2.0, 450, 200)
    KEYFRAME_POSITION("ball", 2.5, 550, 50)
    KEYFRAME_POSITION("ball", 3.0, 650, 200)
    KEYFRAME_POSITION("ball", 3.5, 750, 50)
    KEYFRAME_POSITION("ball", 4.0, 850, 200)
    KEYFRAME_POSITION("ball", 4.5, 950, 50)
    KEYFRAME_POSITION("ball", 5.0, 1050, 200)
    
    # تغيير حجم الكرة (تكبير وتصغير)
    KEYFRAME_SCALE("ball", 0, 1.0)
    KEYFRAME_SCALE("ball", 1.0, 1.5)
    KEYFRAME_SCALE("ball", 2.0, 1.0)
    KEYFRAME_SCALE("ball", 3.0, 1.5)
    KEYFRAME_SCALE("ball", 4.0, 1.0)
    KEYFRAME_SCALE("ball", 5.0, 1.5)
    
    # تغيير لون الكرة (تدرج ألوان قوس قزح)
    KEYFRAME_COLOR("ball", 0, "#FF0000")  # أحمر
    KEYFRAME_COLOR("ball", 0.8, "#FF7F00")  # برتقالي
    KEYFRAME_COLOR("ball", 1.6, "#FFFF00")  # أصفر
    KEYFRAME_COLOR("ball", 2.4, "#00FF00")  # أخضر
    KEYFRAME_COLOR("ball", 3.2, "#0000FF")  # أزرق
    KEYFRAME_COLOR("ball", 4.0, "#4B0082")  # نيلي
    KEYFRAME_COLOR("ball", 4.8, "#9400D3")  # بنفسجي
    KEYFRAME_COLOR("ball", 5.0, "#FF0000")  # عودة للأحمر
    
    # تغيير الشفافية
    KEYFRAME_OPACITY("ball", 0, 1.0)
    KEYFRAME_OPACITY("ball", 2.5, 0.3)
    KEYFRAME_OPACITY("ball", 5.0, 1.0)
    
    # تعيين منحنيات التوقيت
    SET_EASING("ball", "position", "bounce")
    SET_EASING("ball", "scale", "sine")
    SET_EASING("ball", "color", "linear")
    SET_EASING("ball", "opacity", "ease_in_out")
    
    # حفظ التحريك
    SAVE_ANIMATION("bouncing_ball.gif")
    """
    
    plotter.parse_and_execute(commands)
    plotter.plot(title="كرة متحركة مع تغيير اللون والحجم", animate=True)

def example_rotating_shapes():
    """مثال على أشكال متعددة مع دوران وتحولات"""
    plotter = AnimatedPathPlotter()
    
    commands = """
    # إعداد التحريك
    SET_ANIMATION_DURATION(8)
    SET_ANIMATION_FPS(30)
    SET_ANIMATION_OUTPUT("rotating_shapes.gif")
    SET_BACKGROUND_COLOR(#000000)
    
    # إعدادات التحكم
    SHOW_TIMELINE_CONTROLS(true)
    AUTO_PLAY(true)
    LOOP_ANIMATION(true)
    
    # تعريف كائن المربع
    DEFINE_OBJECT("square")
    BEGIN_OBJECT("square")
        PEN_DOWN()
        SET_COLOR_HEX("#FFFFFF")
        SET_THICKNESS(2)
        ENABLE_FILL()
        SET_FILL_COLOR_HEX("#FF0000")
        RECTANGLE(-25, -25, 50, 50)
        PEN_UP()
    END_OBJECT()
    
    # تعريف كائن الدائرة
    DEFINE_OBJECT("circle")
    BEGIN_OBJECT("circle")
        PEN_DOWN()
        SET_COLOR_HEX("#FFFFFF")
        SET_THICKNESS(2)
        ENABLE_FILL()
        SET_FILL_COLOR_HEX("#00FF00")
        CIRCLE(0, 0, 25)
        PEN_UP()
    END_OBJECT()
    
    # تعريف كائن المثلث
    DEFINE_OBJECT("triangle")
    BEGIN_OBJECT("triangle")
        PEN_DOWN()
        SET_COLOR_HEX("#FFFFFF")
        SET_THICKNESS(2)
        ENABLE_FILL()
        SET_FILL_COLOR_HEX("#0000FF")
        MOVE_TO(0, -25)
        LINE_TO(25, 25)
        LINE_TO(-25, 25)
        CLOSE_PATH()
        PEN_UP()
    END_OBJECT()
    
    # تحريك المربع (دوران حول المركز)
    KEYFRAME_POSITION("square", 0, 150, 150)
    KEYFRAME_POSITION("square", 8, 150, 150)
    KEYFRAME_ROTATION("square", 0, 0)
    KEYFRAME_ROTATION("square", 8, 360)
    
    # تحريك الدائرة (حركة دائرية)
    KEYFRAME_POSITION("circle", 0, 250, 150)
    KEYFRAME_POSITION("circle", 2, 350, 250)
    KEYFRAME_POSITION("circle", 4, 250, 350)
    KEYFRAME_POSITION("circle", 6, 150, 250)
    KEYFRAME_POSITION("circle", 8, 250, 150)
    
    # تحريك المثلث (تكبير وتصغير مع دوران)
    KEYFRAME_POSITION("triangle", 0, 350, 350)
    KEYFRAME_POSITION("triangle", 8, 350, 350)
    KEYFRAME_SCALE("triangle", 0, 1.0)
    KEYFRAME_SCALE("triangle", 2, 2.0)
    KEYFRAME_SCALE("triangle", 4, 1.0)
    KEYFRAME_SCALE("triangle", 6, 2.0)
    KEYFRAME_SCALE("triangle", 8, 1.0)
    KEYFRAME_ROTATION("triangle", 0, 0)
    KEYFRAME_ROTATION("triangle", 8, -360)
    
    # تغيير الألوان
    KEYFRAME_FILL_COLOR("square", 0, "#FF0000")
    KEYFRAME_FILL_COLOR("square", 4, "#FF00FF")
    KEYFRAME_FILL_COLOR("square", 8, "#FF0000")
    
    KEYFRAME_FILL_COLOR("circle", 0, "#00FF00")
    KEYFRAME_FILL_COLOR("circle", 4, "#FFFF00")
    KEYFRAME_FILL_COLOR("circle", 8, "#00FF00")
    
    KEYFRAME_FILL_COLOR("triangle", 0, "#0000FF")
    KEYFRAME_FILL_COLOR("triangle", 4, "#00FFFF")
    KEYFRAME_FILL_COLOR("triangle", 8, "#0000FF")
    
    # تعيين منحنيات التوقيت
    SET_EASING("square", "rotation", "linear")
    SET_EASING("circle", "position", "sine")
    SET_EASING("triangle", "scale", "elastic")
    SET_EASING("triangle", "rotation", "linear")
    
    # حفظ التحريك
    SAVE_ANIMATION("rotating_shapes.gif")
    """
    
    plotter.parse_and_execute(commands)
    plotter.plot(title="أشكال متعددة مع دوران وتحولات", animate=True)

def example_character_animation():
    """مثال على تحريك شخصية كرتونية بسيطة"""
    plotter = AnimatedPathPlotter()
    
    commands = """
    # إعداد التحريك
    SET_ANIMATION_DURATION(6)
    SET_ANIMATION_FPS(30)
    SET_ANIMATION_OUTPUT("character_animation.gif")
    SET_BACKGROUND_COLOR(#87CEEB)
    
    # إعدادات التحكم
    SHOW_TIMELINE_CONTROLS(true)
    AUTO_PLAY(true)
    LOOP_ANIMATION(true)
    
    # تعريف كائن الرأس
    DEFINE_OBJECT("head")
    BEGIN_OBJECT("head")
        PEN_DOWN()
        SET_COLOR_HEX("#000000")
        SET_THICKNESS(2)
        ENABLE_FILL()
        SET_FILL_COLOR_HEX("#FFD700")
        CIRCLE(0, 0, 30)
        PEN_UP()
        
        # العين اليمنى
        PEN_DOWN()
        SET_COLOR_HEX("#000000")
        SET_THICKNESS(1)
        ENABLE_FILL()
        SET_FILL_COLOR_HEX("#FFFFFF")
        CIRCLE(-10, 10, 8)
        PEN_UP()
        
        # بؤبؤ العين اليمنى
        PEN_DOWN()
        SET_COLOR_HEX("#000000")
        SET_THICKNESS(1)
        ENABLE_FILL()
        SET_FILL_COLOR_HEX("#000000")
        CIRCLE(-10, 10, 3)
        PEN_UP()
        
        # العين اليسرى
        PEN_DOWN()
        SET_COLOR_HEX("#000000")
        SET_THICKNESS(1)
        ENABLE_FILL()
        SET_FILL_COLOR_HEX("#FFFFFF")
        CIRCLE(10, 10, 8)
        PEN_UP()
        
        # بؤبؤ العين اليسرى
        PEN_DOWN()
        SET_COLOR_HEX("#000000")
        SET_THICKNESS(1)
        ENABLE_FILL()
        SET_FILL_COLOR_HEX("#000000")
        CIRCLE(10, 10, 3)
        PEN_UP()
        
        # الفم
        PEN_DOWN()
        SET_COLOR_HEX("#000000")
        SET_THICKNESS(2)
        MOVE_TO(-15, -10)
        CURVE_TO(-5, -20, 5, -20, 15, -10)
        PEN_UP()
    END_OBJECT()
    
    # تعريف كائن الجسم
    DEFINE_OBJECT("body")
    BEGIN_OBJECT("body")
        PEN_DOWN()
        SET_COLOR_HEX("#000000")
        SET_THICKNESS(2)
        ENABLE_FILL()
        SET_FILL_COLOR_HEX("#FF6347")
        RECTANGLE(-25, 0, 50, 60)
        PEN_UP()
    END_OBJECT()
    
    # تعريف كائن الذراع اليمنى
    DEFINE_OBJECT("right_arm")
    BEGIN_OBJECT("right_arm")
        PEN_DOWN()
        SET_COLOR_HEX("#000000")
        SET_THICKNESS(2)
        ENABLE_FILL()
        SET_FILL_COLOR_HEX("#FF6347")
        RECTANGLE(0, 0, 10, 40)
        PEN_UP()
    END_OBJECT()
    
    # تعريف كائن الذراع اليسرى
    DEFINE_OBJECT("left_arm")
    BEGIN_OBJECT("left_arm")
        PEN_DOWN()
        SET_COLOR_HEX("#000000")
        SET_THICKNESS(2)
        ENABLE_FILL()
        SET_FILL_COLOR_HEX("#FF6347")
        RECTANGLE(0, 0, 10, 40)
        PEN_UP()
    END_OBJECT()
    
    # تعريف كائن الساق اليمنى
    DEFINE_OBJECT("right_leg")
    BEGIN_OBJECT("right_leg")
        PEN_DOWN()
        SET_COLOR_HEX("#000000")
        SET_THICKNESS(2)
        ENABLE_FILL()
        SET_FILL_COLOR_HEX("#1E90FF")
        RECTANGLE(0, 0, 15, 50)
        PEN_UP()
    END_OBJECT()
    
    # تعريف كائن الساق اليسرى
    DEFINE_OBJECT("left_leg")
    BEGIN_OBJECT("left_leg")
        PEN_DOWN()
        SET_COLOR_HEX("#000000")
        SET_THICKNESS(2)
        ENABLE_FILL()
        SET_FILL_COLOR_HEX("#1E90FF")
        RECTANGLE(0, 0, 15, 50)
        PEN_UP()
    END_OBJECT()
    
    # تعيين العلاقات الأبوية
    SET_PARENT("right_arm", "body")
    SET_PARENT("left_arm", "body")
    SET_PARENT("right_leg", "body")
    SET_PARENT("left_leg", "body")
    SET_PARENT("head", "body")
    
    # تحديد مواقع الأجزاء
    KEYFRAME_POSITION("body", 0, 250, 200)
    KEYFRAME_POSITION("head", 0, 250, 140)
    KEYFRAME_POSITION("right_arm", 0, 275, 200)
    KEYFRAME_POSITION("left_arm", 0, 215, 200)
    KEYFRAME_POSITION("right_leg", 0, 260, 260)
    KEYFRAME_POSITION("left_leg", 0, 225, 260)
    
    # تحريك الشخصية (المشي)
    # حركة الجسم
    KEYFRAME_POSITION("body", 0, 100, 200)
    KEYFRAME_POSITION("body", 3, 400, 200)
    KEYFRAME_POSITION("body", 6, 100, 200)
    
    # حركة الرأس (تتبع الجسم)
    KEYFRAME_POSITION("head", 0, 100, 140)
    KEYFRAME_POSITION("head", 3, 400, 140)
    KEYFRAME_POSITION("head", 6, 100, 140)
    
    # حركة الذراعين
    # الذراع اليمنى
    KEYFRAME_POSITION("right_arm", 0, 125, 200)
    KEYFRAME_POSITION("right_arm", 0.75, 125, 200)
    KEYFRAME_POSITION("right_arm", 1.5, 125, 200)
    KEYFRAME_POSITION("right_arm", 2.25, 125, 200)
    KEYFRAME_POSITION("right_arm", 3, 425, 200)
    KEYFRAME_POSITION("right_arm", 3.75, 425, 200)
    KEYFRAME_POSITION("right_arm", 4.5, 425, 200)
    KEYFRAME_POSITION("right_arm", 5.25, 425, 200)
    KEYFRAME_POSITION("right_arm", 6, 125, 200)
    
    KEYFRAME_ROTATION("right_arm", 0, -30)
    KEYFRAME_ROTATION("right_arm", 0.75, 30)
    KEYFRAME_ROTATION("right_arm", 1.5, -30)
    KEYFRAME_ROTATION("right_arm", 2.25, 30)
    KEYFRAME_ROTATION("right_arm", 3, -30)
    KEYFRAME_ROTATION("right_arm", 3.75, 30)
    KEYFRAME_ROTATION("right_arm", 4.5, -30)
    KEYFRAME_ROTATION("right_arm", 5.25, 30)
    KEYFRAME_ROTATION("right_arm", 6, -30)
    
    # الذراع اليسرى
    KEYFRAME_POSITION("left_arm", 0, 65, 200)
    KEYFRAME_POSITION("left_arm", 0.75, 65, 200)
    KEYFRAME_POSITION("left_arm", 1.5, 65, 200)
    KEYFRAME_POSITION("left_arm", 2.25, 65, 200)
    KEYFRAME_POSITION("left_arm", 3, 365, 200)
    KEYFRAME_POSITION("left_arm", 3.75, 365, 200)
    KEYFRAME_POSITION("left_arm", 4.5, 365, 200)
    KEYFRAME_POSITION("left_arm", 5.25, 365, 200)
    KEYFRAME_POSITION("left_arm", 6, 65, 200)
    
    KEYFRAME_ROTATION("left_arm", 0, 30)
    KEYFRAME_ROTATION("left_arm", 0.75, -30)
    KEYFRAME_ROTATION("left_arm", 1.5, 30)
    KEYFRAME_ROTATION("left_arm", 2.25, -30)
    KEYFRAME_ROTATION("left_arm", 3, 30)
    KEYFRAME_ROTATION("left_arm", 3.75, -30)
    KEYFRAME_ROTATION("left_arm", 4.5, 30)
    KEYFRAME_ROTATION("left_arm", 5.25, -30)
    KEYFRAME_ROTATION("left_arm", 6, 30)
    
    # حركة الساقين
    # الساق اليمنى
    KEYFRAME_POSITION("right_leg", 0, 110, 260)
    KEYFRAME_POSITION("right_leg", 0.75, 110, 260)
    KEYFRAME_POSITION("right_leg", 1.5, 110, 260)
    KEYFRAME_POSITION("right_leg", 2.25, 110, 260)
    KEYFRAME_POSITION("right_leg", 3, 410, 260)
    KEYFRAME_POSITION("right_leg", 3.75, 410, 260)
    KEYFRAME_POSITION("right_leg", 4.5, 410, 260)
    KEYFRAME_POSITION("right_leg", 5.25, 410, 260)
    KEYFRAME_POSITION("right_leg", 6, 110, 260)
    
    KEYFRAME_ROTATION("right_leg", 0, 20)
    KEYFRAME_ROTATION("right_leg", 0.75, -20)
    KEYFRAME_ROTATION("right_leg", 1.5, 20)
    KEYFRAME_ROTATION("right_leg", 2.25, -20)
    KEYFRAME_ROTATION("right_leg", 3, 20)
    KEYFRAME_ROTATION("right_leg", 3.75, -20)
    KEYFRAME_ROTATION("right_leg", 4.5, 20)
    KEYFRAME_ROTATION("right_leg", 5.25, -20)
    KEYFRAME_ROTATION("right_leg", 6, 20)
    
    # الساق اليسرى
    KEYFRAME_POSITION("left_leg", 0, 75, 260)
    KEYFRAME_POSITION("left_leg", 0.75, 75, 260)
    KEYFRAME_POSITION("left_leg", 1.5, 75, 260)
    KEYFRAME_POSITION("left_leg", 2.25, 75, 260)
    KEYFRAME_POSITION("left_leg", 3, 375, 260)
    KEYFRAME_POSITION("left_leg", 3.75, 375, 260)
    KEYFRAME_POSITION("left_leg", 4.5, 375, 260)
    KEYFRAME_POSITION("left_leg", 5.25, 375, 260)
    KEYFRAME_POSITION("left_leg", 6, 75, 260)
    
    KEYFRAME_ROTATION("left_leg", 0, -20)
    KEYFRAME_ROTATION("left_leg", 0.75, 20)
    KEYFRAME_ROTATION("left_leg", 1.5, -20)
    KEYFRAME_ROTATION("left_leg", 2.25, 20)
    KEYFRAME_ROTATION("left_leg", 3, -20)
    KEYFRAME_ROTATION("left_leg", 3.75, 20)
    KEYFRAME_ROTATION("left_leg", 4.5, -20)
    KEYFRAME_ROTATION("left_leg", 5.25, 20)
    KEYFRAME_ROTATION("left_leg", 6, -20)
    
    # تعيين منحنيات التوقيت
    SET_EASING("body", "position", "linear")
    SET_EASING("head", "position", "linear")
    SET_EASING("right_arm", "rotation", "sine")
    SET_EASING("left_arm", "rotation", "sine")
    SET_EASING("right_leg", "rotation", "sine")
    SET_EASING("left_leg", "rotation", "sine")
    
    # حفظ التحريك
    SAVE_ANIMATION("character_animation.gif")
    """
    
    plotter.parse_and_execute(commands)
    plotter.plot(title="تحريك شخصية كرتونية بسيطة", animate=True)

def example_color_transformation():
    """مثال على تحريك الألوان والتحولات"""
    plotter = AnimatedPathPlotter()
    
    commands = """
    # إعداد التحريك
    SET_ANIMATION_DURATION(10)
    SET_ANIMATION_FPS(30)
    SET_ANIMATION_OUTPUT("color_transformation.gif")
    SET_BACKGROUND_COLOR(#000000)
    
    # إعدادات التحكم
    SHOW_TIMELINE_CONTROLS(true)
    AUTO_PLAY(true)
    LOOP_ANIMATION(true)
    
    # تعريف كائن النجمة
    DEFINE_OBJECT("star")
    BEGIN_OBJECT("star")
        PEN_DOWN()
        SET_COLOR_HEX("#FFFFFF")
        SET_THICKNESS(2)
        ENABLE_FILL()
        SET_FILL_COLOR_HEX("#FFFF00")
        
        # رسم نجمة خماسية
        MOVE_TO(0, -50)
        LINE_TO(14, 20)
        LINE_TO(47, 20)
        LINE_TO(23, 50)
        LINE_TO(29, 90)
        LINE_TO(0, 70)
        LINE_TO(-29, 90)
        LINE_TO(-23, 50)
        LINE_TO(-47, 20)
        LINE_TO(-14, 20)
        CLOSE_PATH()
        
        PEN_UP()
    END_OBJECT()
    
    # تعريف كائن المربع
    DEFINE_OBJECT("square")
    BEGIN_OBJECT("square")
        PEN_DOWN()
        SET_COLOR_HEX("#FFFFFF")
        SET_THICKNESS(2)
        ENABLE_FILL()
        SET_FILL_COLOR_HEX("#FF0000")
        RECTANGLE(-40, -40, 80, 80)
        PEN_UP()
    END_OBJECT()
    
    # تعريف كائن الدائرة
    DEFINE_OBJECT("circle")
    BEGIN_OBJECT("circle")
        PEN_DOWN()
        SET_COLOR_HEX("#FFFFFF")
        SET_THICKNESS(2)
        ENABLE_FILL()
        SET_FILL_COLOR_HEX("#00FF00")
        CIRCLE(0, 0, 40)
        PEN_UP()
    END_OBJECT()
    
    # تحديد مواقع الأشكال
    KEYFRAME_POSITION("star", 0, 150, 150)
    KEYFRAME_POSITION("square", 0, 400, 150)
    KEYFRAME_POSITION("circle", 0, 650, 150)
    
    # تحريك الألوان
    # النجمة: تغيير لون التعبئة
    KEYFRAME_FILL_COLOR("star", 0, "#FFFF00")  # أصفر
    KEYFRAME_FILL_COLOR("star", 2, "#FF0000")  # أحمر
    KEYFRAME_FILL_COLOR("star", 4, "#00FF00")  # أخضر
    KEYFRAME_FILL_COLOR("star", 6, "#0000FF")  # أزرق
    KEYFRAME_FILL_COLOR("star", 8, "#FF00FF")  # وردي
    KEYFRAME_FILL_COLOR("star", 10, "#FFFF00")  # عودة للأصفر
    
    # المربع: تغيير لون الخط
    KEYFRAME_COLOR("square", 0, "#FFFFFF")  # أبيض
    KEYFRAME_COLOR("square", 2, "#FFFF00")  # أصفر
    KEYFRAME_COLOR("square", 4, "#00FFFF")  # سماوي
    KEYFRAME_COLOR("square", 6, "#FF00FF")  # وردي
    KEYFRAME_COLOR("square", 8, "#00FF00")  # أخضر
    KEYFRAME_COLOR("square", 10, "#FFFFFF")  # عودة للأبيض
    
    # الدائرة: تغيير الشفافية
    KEYFRAME_OPACITY("circle", 0, 1.0)
    KEYFRAME_OPACITY("circle", 2.5, 0.2)
    KEYFRAME_OPACITY("circle", 5, 1.0)
    KEYFRAME_OPACITY("circle", 7.5, 0.2)
    KEYFRAME_OPACITY("circle", 10, 1.0)
    
    # تحريك التحولات
    # النجمة: دوران
    KEYFRAME_ROTATION("star", 0, 0)
    KEYFRAME_ROTATION("star", 10, 360)
    
    # المربع: تكبير وتصغير
    KEYFRAME_SCALE("square", 0, 1.0)
    KEYFRAME_SCALE("square", 2.5, 1.5)
    KEYFRAME_SCALE("square", 5, 0.5)
    KEYFRAME_SCALE("square", 7.5, 1.5)
    KEYFRAME_SCALE("square", 10, 1.0)
    
    # الدائرة: تغيير سماكة الخط
    KEYFRAME_THICKNESS("circle", 0, 2)
    KEYFRAME_THICKNESS("circle", 2.5, 10)
    KEYFRAME_THICKNESS("circle", 5, 2)
    KEYFRAME_THICKNESS("circle", 7.5, 10)
    KEYFRAME_THICKNESS("circle", 10, 2)
    
    # تعيين منحنيات التوقيت
    SET_EASING("star", "fill_color", "linear")
    SET_EASING("star", "rotation", "linear")
    SET_EASING("square", "color", "linear")
    SET_EASING("square", "scale", "bounce")
    SET_EASING("circle", "opacity", "sine")
    SET_EASING("circle", "thickness", "ease_in_out")
    
    # حفظ التحريك
    SAVE_ANIMATION("color_transformation.gif")
    """
    
    plotter.parse_and_execute(commands)
    plotter.plot(title="تحريك الألوان والتحولات", animate=True)

def example_morphing_shapes():
    """مثال على تحويل الأشكال (morphing)"""
    plotter = AnimatedPathPlotter()
    
    commands = """
    # إعداد التحريك
    SET_ANIMATION_DURATION(6)
    SET_ANIMATION_FPS(30)
    SET_ANIMATION_OUTPUT("morphing_shapes.gif")
    SET_BACKGROUND_COLOR(#333333)
    
    # إعدادات التحكم
    SHOW_TIMELINE_CONTROLS(true)
    AUTO_PLAY(true)
    LOOP_ANIMATION(true)
    
    # تعريف كائن المربع
    DEFINE_OBJECT("square")
    BEGIN_OBJECT("square")
        PEN_DOWN()
        SET_COLOR_HEX("#FFFFFF")
        SET_THICKNESS(3)
        ENABLE_FILL()
        SET_FILL_COLOR_HEX("#FF5555")
        RECTANGLE(-40, -40, 80, 80)
        PEN_UP()
    END_OBJECT()
    
    # تعريف كائن الدائرة
    DEFINE_OBJECT("circle")
    BEGIN_OBJECT("circle")
        PEN_DOWN()
        SET_COLOR_HEX("#FFFFFF")
        SET_THICKNESS(3)
        ENABLE_FILL()
        SET_FILL_COLOR_HEX("#55FF55")
        CIRCLE(0, 0, 40)
        PEN_UP()
    END_OBJECT()
    
    # تعريف كائن المثلث
    DEFINE_OBJECT("triangle")
    BEGIN_OBJECT("triangle")
        PEN_DOWN()
        SET_COLOR_HEX("#FFFFFF")
        SET_THICKNESS(3)
        ENABLE_FILL()
        SET_FILL_COLOR_HEX("#5555FF")
        MOVE_TO(0, -40)
        LINE_TO(40, 40)
        LINE_TO(-40, 40)
        CLOSE_PATH()
        PEN_UP()
    END_OBJECT()
    
    # تحديد مواقع الأشكال
    KEYFRAME_POSITION("square", 0, 250, 200)
    KEYFRAME_POSITION("circle", 0, 250, 200)
    KEYFRAME_POSITION("triangle", 0, 250, 200)
    
    # تحديد ظهور الأشكال
    # المربع: ظاهر في البداية ثم يختفي
    KEYFRAME_OPACITY("square", 0, 1.0)
    KEYFRAME_OPACITY("square", 1.8, 1.0)
    KEYFRAME_OPACITY("square", 2.0, 0.0)
    KEYFRAME_OPACITY("square", 4.0, 0.0)
    KEYFRAME_OPACITY("square", 4.2, 1.0)
    KEYFRAME_OPACITY("square", 6.0, 1.0)
    
    # الدائرة: تظهر بعد المربع ثم تختفي
    KEYFRAME_OPACITY("circle", 0, 0.0)
    KEYFRAME_OPACITY("circle", 1.8, 0.0)
    KEYFRAME_OPACITY("circle", 2.0, 1.0)
    KEYFRAME_OPACITY("circle", 3.8, 1.0)
    KEYFRAME_OPACITY("circle", 4.0, 0.0)
    KEYFRAME_OPACITY("circle", 6.0, 0.0)
    
    # المثلث: يظهر بعد الدائرة ثم يختفي
    KEYFRAME_OPACITY("triangle", 0, 0.0)
    KEYFRAME_OPACITY("triangle", 3.8, 0.0)
    KEYFRAME_OPACITY("triangle", 4.0, 1.0)
    KEYFRAME_OPACITY("triangle", 5.8, 1.0)
    KEYFRAME_OPACITY("triangle", 6.0, 0.0)
    
    # تحريك الحجم للانتقال السلس
    # المربع: يكبر ثم يصغر
    KEYFRAME_SCALE("square", 0, 0.2)
    KEYFRAME_SCALE("square", 1.0, 1.0)
    KEYFRAME_SCALE("square", 1.8, 1.2)
    KEYFRAME_SCALE("square", 2.0, 1.5)
    KEYFRAME_SCALE("square", 4.0, 0.2)
    KEYFRAME_SCALE("square", 5.0, 1.0)
    KEYFRAME_SCALE("square", 6.0, 0.2)
    
    # الدائرة: تكبر ثم تصغر
    KEYFRAME_SCALE("circle", 0, 0.2)
    KEYFRAME_SCALE("circle", 1.8, 0.2)
    KEYFRAME_SCALE("circle", 2.0, 0.5)
    KEYFRAME_SCALE("circle", 3.0, 1.0)
    KEYFRAME_SCALE("circle", 3.8, 1.2)
    KEYFRAME_SCALE("circle", 4.0, 1.5)
    KEYFRAME_SCALE("circle", 6.0, 0.2)
    
    # المثلث: يكبر ثم يصغر
    KEYFRAME_SCALE("triangle", 0, 0.2)
    KEYFRAME_SCALE("triangle", 3.8, 0.2)
    KEYFRAME_SCALE("triangle", 4.0, 0.5)
    KEYFRAME_SCALE("triangle", 5.0, 1.0)
    KEYFRAME_SCALE("triangle", 6.0, 0.2)
    
    # تحريك الدوران للانتقال السلس
    KEYFRAME_ROTATION("square", 0, 0)
    KEYFRAME_ROTATION("square", 2.0, 90)
    
    KEYFRAME_ROTATION("circle", 1.8, 90)
    KEYFRAME_ROTATION("circle", 4.0, 180)
    
    KEYFRAME_ROTATION("triangle", 3.8, 180)
    KEYFRAME_ROTATION("triangle", 6.0, 270)
    
    # تغيير الألوان للانتقال السلس
    KEYFRAME_FILL_COLOR("square", 0, "#FF5555")
    KEYFRAME_FILL_COLOR("square", 2.0, "#55FF55")
    
    KEYFRAME_FILL_COLOR("circle", 1.8, "#55FF55")
    KEYFRAME_FILL_COLOR("circle", 4.0, "#5555FF")
    
    KEYFRAME_FILL_COLOR("triangle", 3.8, "#5555FF")
    KEYFRAME_FILL_COLOR("triangle", 6.0, "#FF5555")
    
    # تعيين منحنيات التوقيت
    SET_EASING("square", "opacity", "ease_in_out")
    SET_EASING("square", "scale", "ease_out")
    SET_EASING("square", "rotation", "ease_in_out")
    SET_EASING("square", "fill_color", "linear")
    
    SET_EASING("circle", "opacity", "ease_in_out")
    SET_EASING("circle", "scale", "ease_out")
    SET_EASING("circle", "rotation", "ease_in_out")
    SET_EASING("circle", "fill_color", "linear")
    
    SET_EASING("triangle", "opacity", "ease_in_out")
    SET_EASING("triangle", "scale", "ease_out")
    SET_EASING("triangle", "rotation", "ease_in_out")
    SET_EASING("triangle", "fill_color", "linear")
    
    # حفظ التحريك
    SAVE_ANIMATION("morphing_shapes.gif")
    """
    
    plotter.parse_and_execute(commands)
    plotter.plot(title="تحويل الأشكال (Morphing)", animate=True)

def run_all_examples():
    """تشغيل جميع الأمثلة"""
    examples = [
        example_bouncing_ball,
        example_rotating_shapes,
        example_character_animation,
        example_color_transformation,
        example_morphing_shapes
    ]
    
    for example in examples:
        print(f"\nتشغيل المثال: {example.__name__}")
        print(f"وصف: {example.__doc__}")
        example()

if __name__ == "__main__":
    # تشغيل مثال واحد
    # example_bouncing_ball()
    
    # أو تشغيل جميع الأمثلة
    run_all_examples()
