import unittest
import torch
from torch.nn import functional as F
from vit_lab.model import PatchEmbedding, TinyViT, TinyCNN, count_parameters
from vit_lab.data import SpatialMarkers
from vit_lab.train import make_model, train_one

class VisionTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_patch_shape(self):
        p=PatchEmbedding(3,24,8)
        y,grid=p(torch.randn(3,3,32,32))
        self.assertEqual(y.shape,(3,16,24))
        self.assertEqual(grid,(4,4))

    def test_patch_projection_equals_unfold_plus_linear(self):
        torch.manual_seed(4)
        x=torch.randn(2,3,32,32)
        p=PatchEmbedding(3,24,8)
        actual,_=p(x)
        unfolded=F.unfold(x,kernel_size=8,stride=8).transpose(1,2)
        expected=F.linear(unfolded,p.proj.weight.flatten(1),p.proj.bias)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-5))

    def test_wrong_dimensions_rejected(self):
        p=PatchEmbedding(3,24,8)
        with self.assertRaises(ValueError): p(torch.randn(1,3,33,32))

    def test_vit_forward(self):
        for pooling in ['cls','mean']:
            model=TinyViT(width=24,heads=3,depth=1,pooling=pooling)
            self.assertEqual(model(torch.randn(2,3,32,32)).shape,(2,2))

    def test_positional_interpolation(self):
        model=TinyViT(width=24,heads=3,depth=1)
        p=model.positional_at((6,6))
        self.assertEqual(p.shape,(1,37,24))
        self.assertTrue(torch.equal(p[:,0],model.position[:,0]))
        self.assertEqual(model(torch.randn(2,3,48,48)).shape,(2,2))

    def test_no_position_permutation_invariance(self):
        model=TinyViT(width=24,heads=3,depth=1,use_position=False).eval()
        image=torch.randn(1,3,32,32)
        # Permute 8x8 patches, retaining each patch's internal pixels.
        patches=F.unfold(image,kernel_size=8,stride=8)
        perm=torch.randperm(16)
        shuffled=F.fold(patches[:,:,perm],output_size=(32,32),kernel_size=8,stride=8)
        with torch.no_grad(): a,b=model(image),model(shuffled)
        self.assertTrue(torch.allclose(a,b,atol=1e-5))

    def test_position_sensitive(self):
        model=TinyViT(width=24,heads=3,depth=1,use_position=True)
        self.assertTrue(model.use_position)

    def test_gradients_flow(self):
        model=TinyViT(width=24,heads=3,depth=1)
        model(torch.randn(2,3,32,32)).sum().backward()
        self.assertIsNotNone(model.patch_embed.proj.weight.grad)
        self.assertIsNotNone(model.position.grad)
        self.assertIsNotNone(model.cls_token.grad)

    def test_cnn(self):
        m=TinyCNN()
        self.assertEqual(m(torch.randn(2,3,32,32)).shape,(2,2))
        self.assertGreater(count_parameters(m),100)

    def test_dataset_balance(self):
        ds=SpatialMarkers(100,seed=12)
        self.assertEqual(ds.labels.sum().item(),50)
        self.assertEqual(ds.images.shape,(100,3,32,32))

    def test_dataset_determinism(self):
        a,b=SpatialMarkers(20,seed=12),SpatialMarkers(20,seed=12)
        self.assertTrue(torch.equal(a.images,b.images))
        self.assertTrue(torch.equal(a.labels,b.labels))

    def test_data_separate_seeds(self):
        a,b=SpatialMarkers(20,seed=12),SpatialMarkers(20,seed=13)
        self.assertFalse(torch.equal(a.images,b.images))

    def test_shift_has_same_balanced_labels(self):
        a,b=SpatialMarkers(20,seed=12),SpatialMarkers(20,seed=12,shift=2)
        self.assertTrue(torch.equal(a.labels,b.labels))

    def test_variant_factory(self):
        for n in ('cnn','vit8','vit4','vit8_no_pos','vit8_mean'):
            self.assertEqual(make_model(n)(torch.randn(2,3,32,32)).shape,(2,2))
        with self.assertRaises(ValueError):make_model('wrong')

    def test_tiny_training_smoke(self):
        row=train_one('vit8',seed=0,steps=2,train_count=32,eval_count=16,batch_size=8)
        self.assertTrue(0<=row['clean_accuracy']<=1)
        self.assertEqual(row['patch_tokens'],16)

if __name__ == '__main__':unittest.main()
