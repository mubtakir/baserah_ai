#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
واجهة تصور المعرفة لنظام بصيرة

هذا الملف يحدد واجهة تصور المعرفة لنظام بصيرة، التي تعرض المفاهيم والعلاقات
في شكل رسم بياني تفاعلي، مما يسمح للمستخدمين باستكشاف قاعدة المعرفة بصرياً.

المؤلف: فريق تطوير نظام بصيرة
الإصدار: 1.0.0
"""

import os
import sys
import json
import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_socketio import SocketIO, emit
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# استيراد من الوحدات الأخرى
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cognitive_linguistic_architecture import CognitiveLinguisticArchitecture
from generative_language_model import ConceptualGraph, ConceptNode, ConceptRelation, SemanticVector
from knowledge_extraction_generation import KnowledgeExtractor, KnowledgeGenerator

# تكوين التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('basira_knowledge_visualization')

class KnowledgeVisualizationInterface:
    """واجهة تصور المعرفة لنظام بصيرة."""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8050):
        """
        تهيئة واجهة تصور المعرفة.
        
        Args:
            host: المضيف
            port: المنفذ
        """
        self.host = host
        self.port = port
        
        # تهيئة مكونات النظام
        self.architecture = CognitiveLinguisticArchitecture()
        self.knowledge_extractor = KnowledgeExtractor()
        self.knowledge_generator = KnowledgeGenerator()
        
        # تهيئة الرسم البياني المفاهيمي
        self.conceptual_graph = ConceptualGraph()
        
        # تهيئة تطبيق Dash
        self.app = dash.Dash(__name__, 
                            external_stylesheets=['https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap',
                                                'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css'])
        
        # تكوين تخطيط التطبيق
        self._setup_layout()
        
        # تكوين التفاعلات
        self._setup_callbacks()
        
        # حالة النظام
        self.system_state = {
            'is_initialized': False,
            'active_concept': None,
            'visualization_mode': 'network',  # network, semantic, hierarchical
            'filter_settings': {
                'min_weight': 0.1,
                'relation_types': [],
                'max_depth': 3
            }
        }
    
    def _setup_layout(self):
        """إعداد تخطيط التطبيق."""
        self.app.layout = html.Div([
            # الشريط العلوي
            html.Div([
                html.H1("نظام بصيرة - تصور المعرفة", className="header-title"),
                html.Div([
                    html.Button("تحديث", id="refresh-button", className="control-button"),
                    html.Button("تصدير", id="export-button", className="control-button"),
                    html.Div([
                        html.Label("نمط التصور:", className="control-label"),
                        dcc.Dropdown(
                            id='visualization-mode',
                            options=[
                                {'label': 'شبكة المفاهيم', 'value': 'network'},
                                {'label': 'الفضاء الدلالي', 'value': 'semantic'},
                                {'label': 'التسلسل الهرمي', 'value': 'hierarchical'}
                            ],
                            value='network',
                            clearable=False,
                            className="control-dropdown"
                        )
                    ], className="control-group")
                ], className="header-controls")
            ], className="header"),
            
            # المحتوى الرئيسي
            html.Div([
                # لوحة التحكم الجانبية
                html.Div([
                    html.Div([
                        html.H3("البحث والتصفية", className="sidebar-title"),
                        html.Div([
                            html.Label("بحث عن مفهوم:", className="control-label"),
                            dcc.Input(
                                id="concept-search",
                                type="text",
                                placeholder="أدخل اسم المفهوم...",
                                className="control-input"
                            ),
                            html.Button("بحث", id="search-button", className="control-button")
                        ], className="control-group"),
                        html.Div([
                            html.Label("أنواع العلاقات:", className="control-label"),
                            dcc.Checklist(
                                id="relation-types",
                                options=[
                                    {'label': 'جزء من', 'value': 'part_of'},
                                    {'label': 'نوع من', 'value': 'is_a'},
                                    {'label': 'مرتبط بـ', 'value': 'related_to'},
                                    {'label': 'يؤثر على', 'value': 'affects'},
                                    {'label': 'يتأثر بـ', 'value': 'affected_by'}
                                ],
                                value=['part_of', 'is_a', 'related_to', 'affects', 'affected_by'],
                                className="control-checklist"
                            )
                        ], className="control-group"),
                        html.Div([
                            html.Label("الحد الأدنى للوزن:", className="control-label"),
                            dcc.Slider(
                                id="weight-slider",
                                min=0,
                                max=1,
                                step=0.1,
                                value=0.1,
                                marks={i/10: str(i/10) for i in range(0, 11)},
                                className="control-slider"
                            )
                        ], className="control-group"),
                        html.Div([
                            html.Label("أقصى عمق:", className="control-label"),
                            dcc.Slider(
                                id="depth-slider",
                                min=1,
                                max=5,
                                step=1,
                                value=3,
                                marks={i: str(i) for i in range(1, 6)},
                                className="control-slider"
                            )
                        ], className="control-group")
                    ], className="sidebar-section"),
                    
                    html.Div([
                        html.H3("تفاصيل المفهوم", className="sidebar-title"),
                        html.Div(id="concept-details", className="concept-details")
                    ], className="sidebar-section")
                ], className="sidebar"),
                
                # منطقة الرسم البياني
                html.Div([
                    dcc.Graph(
                        id="knowledge-graph",
                        style={"height": "100%", "width": "100%"},
                        config={'displayModeBar': True}
                    )
                ], className="graph-container")
            ], className="main-content"),
            
            # الشريط السفلي
            html.Div([
                html.Div(id="status-message", className="status-message"),
                html.Div([
                    html.Button("إضافة مفهوم", id="add-concept-button", className="footer-button"),
                    html.Button("إضافة علاقة", id="add-relation-button", className="footer-button"),
                    html.Button("حذف", id="delete-button", className="footer-button")
                ], className="footer-controls")
            ], className="footer"),
            
            # نوافذ الحوار
            html.Div(id="dialog-container", className="dialog-container", style={"display": "none"})
        ], className="app-container")
    
    def _setup_callbacks(self):
        """إعداد التفاعلات."""
        # تحديث الرسم البياني عند تغيير الإعدادات
        @self.app.callback(
            Output("knowledge-graph", "figure"),
            [Input("refresh-button", "n_clicks"),
             Input("visualization-mode", "value"),
             Input("relation-types", "value"),
             Input("weight-slider", "value"),
             Input("depth-slider", "value"),
             Input("search-button", "n_clicks")],
            [State("concept-search", "value")]
        )
        def update_graph(n_clicks, mode, relation_types, min_weight, max_depth, search_clicks, search_term):
            # تحديث إعدادات التصفية
            self.system_state['visualization_mode'] = mode
            self.system_state['filter_settings']['relation_types'] = relation_types
            self.system_state['filter_settings']['min_weight'] = min_weight
            self.system_state['filter_settings']['max_depth'] = max_depth
            
            # البحث عن مفهوم إذا تم تحديده
            if search_term:
                for node_id, node in self.conceptual_graph.nodes.items():
                    if search_term.lower() in node.name.lower():
                        self.system_state['active_concept'] = node_id
                        # تنشيط المفهوم في الرسم البياني
                        self.conceptual_graph.activate_node(node_id, 1.0, True, 0.5, max_depth)
                        break
            
            # إنشاء الرسم البياني المناسب حسب النمط
            if mode == 'network':
                return self._create_network_graph()
            elif mode == 'semantic':
                return self._create_semantic_space()
            elif mode == 'hierarchical':
                return self._create_hierarchical_graph()
            else:
                return self._create_network_graph()
        
        # عرض تفاصيل المفهوم عند النقر على عقدة
        @self.app.callback(
            Output("concept-details", "children"),
            [Input("knowledge-graph", "clickData")]
        )
        def display_concept_details(click_data):
            if not click_data:
                return [html.P("انقر على مفهوم لعرض التفاصيل")]
            
            # استخراج معرف المفهوم من البيانات المنقورة
            point = click_data["points"][0]
            node_id = point.get("customdata", None)
            
            if not node_id or node_id not in self.conceptual_graph.nodes:
                return [html.P("المفهوم غير موجود")]
            
            # الحصول على المفهوم
            node = self.conceptual_graph.nodes[node_id]
            self.system_state['active_concept'] = node_id
            
            # إنشاء عناصر التفاصيل
            details = [
                html.H4(node.name, className="concept-name"),
                html.P(node.description, className="concept-description")
            ]
            
            # إضافة الخصائص
            if node.properties:
                details.append(html.H5("الخصائص:", className="section-title"))
                properties_list = []
                for key, value in node.properties.items():
                    properties_list.append(html.Li(f"{key}: {value}"))
                details.append(html.Ul(properties_list, className="properties-list"))
            
            # إضافة العلاقات
            if node.relations:
                details.append(html.H5("العلاقات:", className="section-title"))
                relations_list = []
                for rel_type, relations in node.relations.items():
                    for relation in relations:
                        target_node = self.conceptual_graph.get_node(relation.target_id)
                        if target_node:
                            relations_list.append(html.Li(f"{rel_type}: {target_node.name} (الوزن: {relation.weight})"))
                details.append(html.Ul(relations_list, className="relations-list"))
            
            # إضافة المتجه الدلالي
            if node.semantic_vector:
                details.append(html.H5("المتجه الدلالي:", className="section-title"))
                semantic_values = []
                for dim, value in node.semantic_vector.dimensions.items():
                    semantic_values.append(html.Li(f"{dim.name}: {value:.2f}"))
                details.append(html.Ul(semantic_values, className="semantic-list"))
            
            return details
        
        # فتح نافذة حوار إضافة مفهوم
        @self.app.callback(
            Output("dialog-container", "children"),
            [Input("add-concept-button", "n_clicks"),
             Input("add-relation-button", "n_clicks")],
            [State("dialog-container", "children")]
        )
        def open_dialog(add_concept_clicks, add_relation_clicks, current_children):
            ctx = dash.callback_context
            if not ctx.triggered:
                return current_children
            
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if button_id == "add-concept-button":
                return self._create_add_concept_dialog()
            elif button_id == "add-relation-button":
                return self._create_add_relation_dialog()
            
            return current_children
        
        # تغيير حالة عرض نافذة الحوار
        @self.app.callback(
            Output("dialog-container", "style"),
            [Input("dialog-container", "children"),
             Input("dialog-close", "n_clicks")]
        )
        def toggle_dialog(children, close_clicks):
            ctx = dash.callback_context
            if not ctx.triggered:
                return {"display": "none"}
            
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if button_id == "dialog-container" and children:
                return {"display": "flex"}
            elif button_id == "dialog-close":
                return {"display": "none"}
            
            return {"display": "none"}
        
        # إضافة مفهوم جديد
        @self.app.callback(
            Output("status-message", "children"),
            [Input("add-concept-submit", "n_clicks")],
            [State("concept-name", "value"),
             State("concept-description", "value")]
        )
        def add_new_concept(n_clicks, name, description):
            if not n_clicks or not name:
                return ""
            
            # إنشاء معرف فريد للمفهوم
            node_id = f"concept_{len(self.conceptual_graph.nodes) + 1}"
            
            # إنشاء متجه دلالي عشوائي (يمكن تحسينه لاحقاً)
            semantic_vector = SemanticVector(
                values=np.random.rand(10),
                source="user_input"
            )
            
            # إنشاء عقدة المفهوم
            node = ConceptNode(
                id=node_id,
                name=name,
                description=description or "",
                semantic_vector=semantic_vector
            )
            
            # إضافة المفهوم إلى الرسم البياني
            self.conceptual_graph.add_node(node)
            
            return f"تمت إضافة المفهوم: {name}"
        
        # إضافة علاقة جديدة
        @self.app.callback(
            Output("status-message", "children", allow_duplicate=True),
            [Input("add-relation-submit", "n_clicks")],
            [State("source-concept", "value"),
             State("target-concept", "value"),
             State("relation-type", "value"),
             State("relation-weight", "value")]
        )
        def add_new_relation(n_clicks, source_id, target_id, relation_type, weight):
            if not n_clicks or not source_id or not target_id or not relation_type:
                return ""
            
            # التحقق من وجود المفاهيم
            if source_id not in self.conceptual_graph.nodes or target_id not in self.conceptual_graph.nodes:
                return "المفاهيم غير موجودة"
            
            # إنشاء العلاقة
            relation = ConceptRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                weight=weight or 1.0
            )
            
            # إضافة العلاقة إلى الرسم البياني
            success = self.conceptual_graph.add_relation(relation)
            
            if success:
                source_name = self.conceptual_graph.nodes[source_id].name
                target_name = self.conceptual_graph.nodes[target_id].name
                return f"تمت إضافة العلاقة: {source_name} {relation_type} {target_name}"
            else:
                return "فشل في إضافة العلاقة"
        
        # حذف مفهوم أو علاقة
        @self.app.callback(
            Output("status-message", "children", allow_duplicate=True),
            [Input("delete-button", "n_clicks")],
            [State("knowledge-graph", "clickData")]
        )
        def delete_element(n_clicks, click_data):
            if not n_clicks or not click_data:
                return ""
            
            # استخراج معرف المفهوم من البيانات المنقورة
            point = click_data["points"][0]
            node_id = point.get("customdata", None)
            
            if not node_id or node_id not in self.conceptual_graph.nodes:
                return "لم يتم تحديد عنصر للحذف"
            
            # حذف المفهوم
            node_name = self.conceptual_graph.nodes[node_id].name
            del self.conceptual_graph.nodes[node_id]
            
            # حذف العلاقات المرتبطة
            for node in self.conceptual_graph.nodes.values():
                for rel_type in list(node.relations.keys()):
                    node.relations[rel_type] = [rel for rel in node.relations[rel_type] if rel.target_id != node_id]
            
            return f"تم حذف المفهوم: {node_name}"
    
    def _create_network_graph(self) -> go.Figure:
        """
        إنشاء رسم بياني شبكي للمفاهيم.
        
        Returns:
            رسم بياني Plotly
        """
        # تحويل الرسم البياني المفاهيمي إلى رسم بياني NetworkX
        G = nx.DiGraph()
        
        # إضافة العقد
        for node_id, node in self.conceptual_graph.nodes.items():
            G.add_node(node_id, name=node.name, activation=node.activation)
        
        # إضافة الحواف
        edges = []
        for node_id, node in self.conceptual_graph.nodes.items():
            for rel_type, relations in node.relations.items():
                # تصفية العلاقات حسب النوع والوزن
                if rel_type in self.system_state['filter_settings']['relation_types']:
                    for relation in relations:
                        if relation.weight >= self.system_state['filter_settings']['min_weight']:
                            G.add_edge(node_id, relation.target_id, 
                                      weight=relation.weight, 
                                      relation_type=rel_type)
                            edges.append((node_id, relation.target_id, relation.weight, rel_type))
        
        # حساب تخطيط الرسم البياني
        pos = nx.spring_layout(G, seed=42)
        
        # إنشاء رسم بياني Plotly
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in edges:
            source, target, weight, rel_type = edge
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(f"{rel_type} ({weight:.2f})")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines'
        )
        
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        node_ids = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_info = G.nodes[node]
            node_text.append(node_info['name'])
            
            # حجم العقدة يعتمد على التنشيط
            activation = node_info.get('activation', 0)
            node_size.append(20 + 30 * activation)
            
            # لون العقدة يعتمد على التنشيط
            node_color.append(activation)
            
            node_ids.append(node)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlOrRd',
                color=node_color,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title='مستوى التنشيط',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2)
            ),
            customdata=node_ids
        )
        
        # إنشاء الرسم البياني
        fig = go.Figure(data=[edge_trace, node_trace],
                      layout=go.Layout(
                          title='شبكة المفاهيم',
                          titlefont=dict(size=16),
                          showlegend=False,
                          hovermode='closest',
                          margin=dict(b=20, l=5, r=5, t=40),
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          plot_bgcolor='rgba(248,248,248,1)',
                          paper_bgcolor='rgba(248,248,248,1)',
                          annotations=[
                              dict(
                                  ax=edge_x[i], ay=edge_y[i],
                                  axref='x', ayref='y',
                                  x=edge_x[i+1], y=edge_y[i+1],
                                  xref='x', yref='y',
                                  showarrow=True,
                                  arrowhead=3,
                                  arrowsize=1.5,
                                  arrowwidth=1.5,
                                  opacity=0.7
                              ) for i in range(0, len(edge_x)-1, 3)
                          ]
                      ))
        
        return fig
    
    def _create_semantic_space(self) -> go.Figure:
        """
        إنشاء فضاء دلالي للمفاهيم.
        
        Returns:
            رسم بياني Plotly
        """
        # جمع المتجهات الدلالية للمفاهيم
        semantic_vectors = []
        node_names = []
        node_ids = []
        node_activations = []
        
        for node_id, node in self.conceptual_graph.nodes.items():
            if node.semantic_vector is not None:
                semantic_vectors.append(node.semantic_vector.values)
                node_names.append(node.name)
                node_ids.append(node_id)
                node_activations.append(node.activation)
        
        if not semantic_vectors:
            # إنشاء رسم بياني فارغ إذا لم تكن هناك متجهات دلالية
            return go.Figure(layout=go.Layout(
                title='الفضاء الدلالي (لا توجد متجهات دلالية)',
                titlefont=dict(size=16),
                showlegend=False,
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=True, zeroline=True, showticklabels=True),
                yaxis=dict(showgrid=True, zeroline=True, showticklabels=True),
                plot_bgcolor='rgba(248,248,248,1)',
                paper_bgcolor='rgba(248,248,248,1)'
            ))
        
        # تحويل المتجهات إلى مصفوفة NumPy
        vectors = np.array(semantic_vectors)
        
        # تقليل الأبعاد إلى 2D باستخدام PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors)
        
        # إنشاء رسم بياني Plotly
        fig = go.Figure()
        
        # إضافة النقاط
        fig.add_trace(go.Scatter(
            x=vectors_2d[:, 0],
            y=vectors_2d[:, 1],
            mode='markers+text',
            marker=dict(
                size=[20 + 30 * act for act in node_activations],
                color=node_activations,
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(
                    thickness=15,
                    title='مستوى التنشيط',
                    xanchor='left',
                    titleside='right'
                )
            ),
            text=node_names,
            textposition='top center',
            hoverinfo='text',
            customdata=node_ids
        ))
        
        # تكوين التخطيط
        fig.update_layout(
            title='الفضاء الدلالي للمفاهيم',
            titlefont=dict(size=16),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(
                title=f'المكون الرئيسي 1 ({pca.explained_variance_ratio_[0]:.2%})',
                showgrid=True,
                zeroline=True,
                showticklabels=True
            ),
            yaxis=dict(
                title=f'المكون الرئيسي 2 ({pca.explained_variance_ratio_[1]:.2%})',
                showgrid=True,
                zeroline=True,
                showticklabels=True
            ),
            plot_bgcolor='rgba(248,248,248,1)',
            paper_bgcolor='rgba(248,248,248,1)'
        )
        
        return fig
    
    def _create_hierarchical_graph(self) -> go.Figure:
        """
        إنشاء رسم بياني هرمي للمفاهيم.
        
        Returns:
            رسم بياني Plotly
        """
        # إنشاء قاموس للعلاقات الهرمية
        hierarchy = {}
        
        # البحث عن علاقات "نوع من" و"جزء من"
        for node_id, node in self.conceptual_graph.nodes.items():
            for rel_type, relations in node.relations.items():
                if rel_type in ['is_a', 'part_of'] and rel_type in self.system_state['filter_settings']['relation_types']:
                    for relation in relations:
                        if relation.weight >= self.system_state['filter_settings']['min_weight']:
                            if node_id not in hierarchy:
                                hierarchy[node_id] = []
                            hierarchy[node_id].append((relation.target_id, rel_type, relation.weight))
        
        # إنشاء بيانات الشجرة
        labels = {}
        parents = {}
        values = {}
        colors = {}
        ids = {}
        
        for node_id, node in self.conceptual_graph.nodes.items():
            labels[node_id] = node.name
            values[node_id] = 1
            colors[node_id] = node.activation
            ids[node_id] = node_id
            
            if node_id in hierarchy:
                for target_id, rel_type, weight in hierarchy[node_id]:
                    parents[node_id] = target_id
            else:
                parents[node_id] = ""
        
        # إنشاء رسم بياني Plotly
        fig = go.Figure(go.Sunburst(
            ids=list(ids.values()),
            labels=list(labels.values()),
            parents=list(parents.values()),
            values=list(values.values()),
            branchvalues="total",
            marker=dict(
                colors=list(colors.values()),
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(
                    thickness=15,
                    title='مستوى التنشيط',
                    xanchor='left',
                    titleside='right'
                )
            ),
            hovertemplate='<b>%{label}</b><br>المستوى: %{color:.2f}<extra></extra>',
            customdata=list(ids.values())
        ))
        
        # تكوين التخطيط
        fig.update_layout(
            title='التسلسل الهرمي للمفاهيم',
            titlefont=dict(size=16),
            margin=dict(t=60, l=0, r=0, b=0)
        )
        
        return fig
    
    def _create_add_concept_dialog(self) -> html.Div:
        """
        إنشاء نافذة حوار إضافة مفهوم.
        
        Returns:
            عنصر Div يحتوي على نافذة الحوار
        """
        return html.Div([
            html.Div([
                html.H3("إضافة مفهوم جديد", className="dialog-title"),
                html.Button("×", id="dialog-close", className="dialog-close")
            ], className="dialog-header"),
            html.Div([
                html.Div([
                    html.Label("اسم المفهوم:", className="dialog-label"),
                    dcc.Input(id="concept-name", type="text", className="dialog-input")
                ], className="dialog-group"),
                html.Div([
                    html.Label("وصف المفهوم:", className="dialog-label"),
                    dcc.Textarea(id="concept-description", className="dialog-textarea")
                ], className="dialog-group")
            ], className="dialog-body"),
            html.Div([
                html.Button("إلغاء", id="dialog-close", className="dialog-button"),
                html.Button("إضافة", id="add-concept-submit", className="dialog-button dialog-submit")
            ], className="dialog-footer")
        ], className="dialog")
    
    def _create_add_relation_dialog(self) -> html.Div:
        """
        إنشاء نافذة حوار إضافة علاقة.
        
        Returns:
            عنصر Div يحتوي على نافذة الحوار
        """
        # إنشاء قائمة بالمفاهيم المتاحة
        concept_options = [{'label': node.name, 'value': node_id} 
                          for node_id, node in self.conceptual_graph.nodes.items()]
        
        return html.Div([
            html.Div([
                html.H3("إضافة علاقة جديدة", className="dialog-title"),
                html.Button("×", id="dialog-close", className="dialog-close")
            ], className="dialog-header"),
            html.Div([
                html.Div([
                    html.Label("المفهوم المصدر:", className="dialog-label"),
                    dcc.Dropdown(
                        id="source-concept",
                        options=concept_options,
                        className="dialog-dropdown"
                    )
                ], className="dialog-group"),
                html.Div([
                    html.Label("نوع العلاقة:", className="dialog-label"),
                    dcc.Dropdown(
                        id="relation-type",
                        options=[
                            {'label': 'جزء من', 'value': 'part_of'},
                            {'label': 'نوع من', 'value': 'is_a'},
                            {'label': 'مرتبط بـ', 'value': 'related_to'},
                            {'label': 'يؤثر على', 'value': 'affects'},
                            {'label': 'يتأثر بـ', 'value': 'affected_by'}
                        ],
                        className="dialog-dropdown"
                    )
                ], className="dialog-group"),
                html.Div([
                    html.Label("المفهوم الهدف:", className="dialog-label"),
                    dcc.Dropdown(
                        id="target-concept",
                        options=concept_options,
                        className="dialog-dropdown"
                    )
                ], className="dialog-group"),
                html.Div([
                    html.Label("وزن العلاقة:", className="dialog-label"),
                    dcc.Slider(
                        id="relation-weight",
                        min=0,
                        max=1,
                        step=0.1,
                        value=1.0,
                        marks={i/10: str(i/10) for i in range(0, 11)},
                        className="dialog-slider"
                    )
                ], className="dialog-group")
            ], className="dialog-body"),
            html.Div([
                html.Button("إلغاء", id="dialog-close", className="dialog-button"),
                html.Button("إضافة", id="add-relation-submit", className="dialog-button dialog-submit")
            ], className="dialog-footer")
        ], className="dialog")
    
    def load_sample_data(self):
        """تحميل بيانات عينة للعرض التوضيحي."""
        # إنشاء بعض المفاهيم
        concepts = [
            ("concept_1", "اللغة", "نظام من الرموز الصوتية أو المكتوبة يستخدم للتواصل"),
            ("concept_2", "اللغة العربية", "إحدى اللغات السامية، وهي لغة القرآن الكريم"),
            ("concept_3", "النحو", "علم يبحث في أصول تكوين الجملة وقواعد الإعراب"),
            ("concept_4", "الصرف", "علم يبحث في بنية الكلمة وصيغها وأوزانها"),
            ("concept_5", "البلاغة", "علم يبحث في أساليب الكلام وجماله وتأثيره"),
            ("concept_6", "الجملة", "وحدة لغوية تتكون من كلمات مترابطة تعبر عن معنى تام"),
            ("concept_7", "الكلمة", "وحدة لغوية ذات معنى مستقل"),
            ("concept_8", "الحرف", "أصغر وحدة صوتية في اللغة")
        ]
        
        # إضافة المفاهيم
        for concept_id, name, description in concepts:
            # إنشاء متجه دلالي عشوائي
            semantic_vector = SemanticVector(
                values=np.random.rand(10),
                source="sample_data"
            )
            
            # إنشاء عقدة المفهوم
            node = ConceptNode(
                id=concept_id,
                name=name,
                description=description,
                semantic_vector=semantic_vector
            )
            
            # إضافة المفهوم إلى الرسم البياني
            self.conceptual_graph.add_node(node)
        
        # إضافة العلاقات
        relations = [
            ("concept_2", "concept_1", "is_a", 1.0),  # اللغة العربية نوع من اللغة
            ("concept_3", "concept_2", "part_of", 0.9),  # النحو جزء من اللغة العربية
            ("concept_4", "concept_2", "part_of", 0.9),  # الصرف جزء من اللغة العربية
            ("concept_5", "concept_2", "part_of", 0.8),  # البلاغة جزء من اللغة العربية
            ("concept_6", "concept_2", "part_of", 0.7),  # الجملة جزء من اللغة العربية
            ("concept_7", "concept_6", "part_of", 0.9),  # الكلمة جزء من الجملة
            ("concept_8", "concept_7", "part_of", 0.9),  # الحرف جزء من الكلمة
            ("concept_3", "concept_6", "affects", 0.8),  # النحو يؤثر على الجملة
            ("concept_4", "concept_7", "affects", 0.8),  # الصرف يؤثر على الكلمة
            ("concept_5", "concept_6", "affects", 0.7)   # البلاغة تؤثر على الجملة
        ]
        
        # إضافة العلاقات
        for source_id, target_id, relation_type, weight in relations:
            relation = ConceptRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                weight=weight
            )
            
            self.conceptual_graph.add_relation(relation)
    
    def run(self):
        """تشغيل واجهة تصور المعرفة."""
        # تحميل بيانات عينة
        self.load_sample_data()
        
        # تعيين حالة النظام
        self.system_state['is_initialized'] = True
        
        # تشغيل التطبيق
        self.app.run_server(host=self.host, port=self.port, debug=True)


# تشغيل التطبيق إذا تم تنفيذ الملف مباشرة
if __name__ == '__main__':
    interface = KnowledgeVisualizationInterface()
    interface.run()
