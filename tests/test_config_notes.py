"""配置说明不改变运行参数，也不能削弱实体配置指纹。"""
import json
from pathlib import Path
import tempfile
import unittest

from ur5e_sim.config import load_settings, scene_signature
from ur5e_sim.config_io import read_config


class ConfigNotesTests(unittest.TestCase):
    def test_notes_ignored_but_geometry_changes_detected(self):
        settings = load_settings()
        expected = scene_signature(settings)
        with tempfile.TemporaryDirectory() as directory:
            for field in ('camera_config', 'port_config'):
                original = Path(settings[field])
                data = read_config(original)
                if field == 'port_config':
                    data['model']['path'] = str((original.parent / data['model']['path']).resolve())
                    data['render']['material_regions'][0]['_说明'] = {'gray': '仅改说明'}
                    data['port']['_说明'] = {'center': '中文说明'}
                else:
                    data['output_profile']['intrinsics_nominal']['_说明'] = {'fx': '像素'}
                data['_说明'] = {'用途': '不会参与配置指纹'}
                target = Path(directory) / original.name
                target.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
                settings[field] = str(target)
            settings['_说明'] = {'fixture_height_m': '支撑台高度'}
            self.assertEqual(scene_signature(settings), expected)
            port_path = Path(settings['port_config'])
            clean = read_config(port_path)
            self.assertNotIn('_说明', clean['render']['material_regions'][0])
            clean['port']['center'][0] += 1
            port_path.write_text(json.dumps(clean), encoding='utf-8')
            self.assertNotEqual(scene_signature(settings), expected)
