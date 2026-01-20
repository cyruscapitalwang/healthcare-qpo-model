# Healthcare QP Clinical Prediction Model (uv)

This project demonstrates a **healthcare-style prediction model** with:

- Qualifying Population (QP)
- Cut-off date
- Out-of-Time (OOT) validation
- Target population defined in a future window

## Concepts
- **QP**: Members eligible at the cutoff date
- **Cutoff date**: Freezes what is known vs what is predicted
- **Target**: Future high-cost indicator
- **OOT**: Later cutoff date to simulate production performance

## Run
```bash
uv run healthcare-qp-model

