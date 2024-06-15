# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules 
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

hidden_imports = collect_submodules('PIL')
hidden_imports += collect_submodules('PyQt5')
import os
import sys

project_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
datas = collect_data_files('assets', includes=['*.ico'])

a = Analysis(['src\\main.py'],
             pathex=[project_dir, os.path.join(project_dir, 'src')], 
             binaries=[],
             datas=datas,  # Include all filesfrom the 'data' folder
             hiddenimports=hidden_imports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas, 
          [],
          name='Multiband Imaging',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          icon='icon.ico'
)
