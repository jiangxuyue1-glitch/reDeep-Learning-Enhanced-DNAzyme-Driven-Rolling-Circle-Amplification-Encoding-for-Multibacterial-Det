from bactorcanet.cli import main

main(["evaluate", "--checkpoint", "runs/default/best.pt", "--data-dir", "data", "--split", "test", "--output-dir", "reports/test"])
