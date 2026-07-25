# CATENA 4-GPU full-access 연구계획

- `main.tex`: ko.TeX/XeLaTeX 원고
- `CATENA_REALM2026_fullaccess_4gpu_research_plan_v4_ko.pdf`: 컴파일된 연구계획서

E00 PASS 실측 결과와 일정 변동은 `main.tex`에 반영돼 있다. 현재 host에는
TeX engine이 없어 PDF는 갱신 전 버전이며, 제출용으로 사용하기 전에 아래
명령으로 다시 빌드하고 checksum을 갱신해야 한다.

컴파일:

```bash
latexmk -xelatex -interaction=nonstopmode main.tex
```
