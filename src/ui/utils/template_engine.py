"""
Jinja2 template engine setup.

Példa használat:
    from ui.utils import render
    html = render('header.html', title='My App', icon='🎮')
"""

from jinja2 import Environment, FileSystemLoader, select_autoescape
import os


def get_jinja_env():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template_dir = os.path.join(base_dir, 'templates')

    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(['html', 'xml']),
        trim_blocks=True,
        lstrip_blocks=True
    )

    return env


jinja_env = get_jinja_env()


def render(template_name, **context):
    template = jinja_env.get_template(template_name)
    return template.render(**context)