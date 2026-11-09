# 실행 가이드 (Windows)

이 프로젝트는 Python 3.11 가상환경(`.venv`)을 사용합니다.
`run_gui.cmd`를 실행하면 올바른 인터프리터로 GUI가 시작됩니다.

## 빠른 실행

PowerShell에서:

```powershell
cd C:\Users\fall2\work\mp3_study_app
.\run_gui.cmd
```

탐색기에서 `run_gui.cmd`를 더블클릭해도 됩니다.

옵션을 넘길 때:

```powershell
.\run_gui.cmd --theme dark
.\run_gui.cmd --project data\project\ksat_2026_english
```

## 왜 `python gui.py`가 아니라 `run_gui.cmd`인가요?

이 PC에서는 터미널의 `python`이 MSYS2 Python 3.14를 먼저 잡을 수 있습니다.
그 인터프리터에는 `customtkinter`, `torch`, `openai-whisper` 등이 없어서
다음 오류가 납니다.

```
ModuleNotFoundError: No module named 'customtkinter'
```

프로젝트 의존성은 Python 3.11에 설치되어 있고, `.venv`도 Python 3.11로
만들었습니다. `run_gui.cmd`는 항상 `.venv`의 Python을 사용하며,
`.venv`가 없으면 `py -3.11`로 자동 대체합니다.

## 수동으로 실행하기

가상환경을 활성화한 뒤 실행:

```powershell
.\.venv\Scripts\Activate.ps1
python gui.py
```

활성화 없이 바로 실행:

```powershell
.\.venv\Scripts\python.exe gui.py
```

PowerShell에서 `Activate.ps1`이 차단된다면:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 환경 확인

```powershell
.\.venv\Scripts\python.exe -c "import sys, customtkinter, torch, pygame; print(sys.version); print(customtkinter.__version__); print(torch.__version__)"
```

다음이 출력되면 정상입니다.

- Python `3.11.x`
- `customtkinter 6.0.0`
- `torch 2.15.0+xpu` (설치된 버전에 따라 다름)

## 참고

- `.venv/`는 `.gitignore`에 포함되어 있어 커밋되지 않습니다.
- `.venv`는 `--system-site-packages`로 만들어서 기존 Python 3.11의 전역
  패키지를 그대로 재사용합니다. 패키지를 추가로 설치하려면:

  ```powershell
  .\.venv\Scripts\python.exe -m pip install <패키지>
  ```

- 완전히 독립된 환경을 원하면 `.venv`를 지우고 아래처럼 다시 만든 뒤
  `requirements.txt`를 설치하세요.

  ```powershell
  Remove-Item -Recurse -Force .venv
  py -3.11 -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  ```

## 문제 해결

| 증상 | 해결 |
|---|---|
| `ModuleNotFoundError: customtkinter` | `run_gui.cmd`로 실행했는지 확인 |
| `Activate.ps1 cannot be loaded` | 위의 `Set-ExecutionPolicy -Scope Process` 실행 |
| `py -3.11`을 찾을 수 없음 | Python 3.11 설치 또는 `run_gui.cmd`의 경로 수정 |
| GPU/Whisper 관련 오류 | `README.md`의 설치/문제 해결 절 참고 |
