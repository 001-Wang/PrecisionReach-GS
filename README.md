# PrecisionReach-GS

Project page for **Robot-Aligned 3D Gaussian Splatting for Precise Reaching with Wrist-Camera VLA Policies**.

PrecisionReach-GS is a lightweight real-to-sim-to-real pipeline for data-efficient VLA adaptation. It reconstructs a real workspace with 3D Gaussian Splatting, aligns the representation with the robot frame, and generates synchronized wrist-camera observations and smooth reaching actions.

## Project page

[https://001-wang.github.io/PrecisionReach-GS/](https://001-wang.github.io/PrecisionReach-GS/)

## Highlights

- Wrist-camera-only VLA deployment for precise pre-contact reaching
- Up to 1,500 generated trajectories from short real-world captures
- Less than 10 minutes of active human capture for the largest generated dataset
- Real-robot evaluation on screw, GPU, and RAM targets after workspace repositioning

## Local preview

The site is plain HTML, CSS, and JavaScript. Serve the repository root with any static web server, for example:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Citation

```bibtex
@unpublished{wang_precisionreach_gs,
  title  = {Robot-Aligned 3D Gaussian Splatting for Precise Reaching
            with Wrist-Camera VLA Policies},
  author = {Wang, Zuoxu},
  note   = {Manuscript under review},
  url    = {https://github.com/001-Wang/PrecisionReach-GS}
}
```

The research implementation will be added when it is ready for release.
