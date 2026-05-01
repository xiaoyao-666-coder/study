import torch
from gccl_model import ImageGraphConv, build_k_hop_adj_matrix, GCCL_Model


def test_image_graph_conv():
    """测试 ImageGraphConv"""
    print("Testing ImageGraphConv...")
    batch_size, channels, height, width = 2, 1, 8, 8
    num_nodes = height * width
    x_t = torch.randn(batch_size, channels, height, width)
    adj = torch.eye(num_nodes)
    graph_conv = ImageGraphConv(channels, 16)
    gc_t = graph_conv(x_t, adj)
    print(f"  Input: {x_t.shape} -> Output: {gc_t.shape}")
    assert gc_t.shape == (batch_size, 16, height, width), "ImageGraphConv output shape error"
    print("  [OK] ImageGraphConv test passed")
    return True


def test_build_adj_matrix():
    """测试 build_k_hop_adj_matrix"""
    print("\nTesting build_k_hop_adj_matrix...")
    train_data = torch.randn(100, 8, 8)
    adj = build_k_hop_adj_matrix(train_data, theta=3, k=2)
    print(f"  Adjacency matrix shape: {adj.shape}")
    assert adj.shape == (64, 64), "Adjacency matrix shape error"
    print("  [OK] build_k_hop_adj_matrix test passed")
    return True


def test_gccl_model():
    """测试 GCCL_Model"""
    print("\nTesting GCCL_Model...")
    model = GCCL_Model(
        input_dim=1,
        gc_dim=8,
        hidden_dim_g=16,
        hidden_dim_c=16,
        kernel_size=(3, 3),
        img_height=8,
        img_width=8,
        theta=3,
        k_hop=2,
        output_dim=1,
    )
    frames = torch.randn(2, 5, 8, 8, 1)
    adj = build_k_hop_adj_matrix(torch.randn(50, 8, 8), theta=3, k=2)
    next_frames = model(frames, adj, num_output_frames=2)
    print(f"  Input: {frames.shape} -> Output: {next_frames.shape}")
    assert next_frames.shape == (2, 2, 8, 8, 1), "GCCL_Model output shape error"
    print("  [OK] GCCL_Model test passed")
    return True


def test_predict():
    """测试 predict 方法"""
    print("\nTesting predict method...")
    model = GCCL_Model(
        input_dim=1,
        gc_dim=8,
        hidden_dim_g=16,
        hidden_dim_c=16,
        kernel_size=(3, 3),
        img_height=8,
        img_width=8,
        theta=3,
        k_hop=2,
        output_dim=1,
    )
    frames = torch.randn(2, 3, 8, 8, 1)
    adj = build_k_hop_adj_matrix(torch.randn(50, 8, 8), theta=3, k=2)
    predictions = model.predict(frames, adj, num_output_frames=2)
    print(f"  Prediction output: {predictions.shape}")
    assert predictions.shape == (2, 2, 8, 8, 1), "predict output shape error"
    print("  [OK] predict test passed")
    return True


if __name__ == "__main__":
    try:
        test_image_graph_conv()
        test_build_adj_matrix()
        test_gccl_model()
        test_predict()
        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\n[Test Failed] {e}")
        import traceback
        traceback.print_exc()
