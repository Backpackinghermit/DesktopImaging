# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules
block_cipher = None

hidden_imports = collect_submodules('PIL')
hidden_imports += collect_submodules('PyQt5')
import os
import sys

project_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

a = Analysis(['src\\main.py'],
             pathex=[project_dir, os.path.join(project_dir, 'src')], 
             binaries=[],
             datas=[],  # Include all filesfrom the 'data' folder
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
          name='your_script_name',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False )
