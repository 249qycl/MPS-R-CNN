from solver.sparse_rcnn_solver import SparseRCNNSolver

if __name__ == '__main__':
    processor = SparseRCNNSolver(cfg_path="config/sparse_rcnn.yaml")
    processor.run()
