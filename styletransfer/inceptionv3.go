package styletransfer

import (
	"fmt"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/models/inceptionv3"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/janpfeifer/must"
)

// This file holds InceptionV3 specific functions.

// InceptionV3Resize resizes the image(s) to the size used by InceptionV3 by
// using interpolation.
func InceptionV3Resize(img *Node) *Node {
	isSingle := img.Rank() == 3
	if isSingle {
		// Add batch dimension.
		img = ExpandAxes(img, 0)
	}
	size := inceptionv3.ClassificationImageSize
	img = Interpolate(img, 1, size, size, 3).Done()
	if isSingle {
		// Remove batch dimension.
		img = Squeeze(img, 0)
	}
	return img
}

// InceptionV3ResizeTensor is a small wrapper on InceptionV3Resize, but it works on materialized tensors.
func InceptionV3ResizeTensor(backend backends.Backend, img *tensors.Tensor) *tensors.Tensor {
	return ExecOnce(backend, func(img *Node) *Node {
		return InceptionV3Resize(img)
	}, img)
}

var (
	// InceptionV3Dir is where to cache the InceptionV3 model weights.
	// They will be downloaded there the first time it runs.
	InceptionV3Dir = "~/.cache/inceptionv3"

	// InceptionV3NumLayers is the number of layers exported by InceptionV3 model.
	InceptionV3NumLayers = 93
)

// InceptionV3PerLayerEmbeddings creates the embeddings for each of the images using a frozen InceptionV3
// model.
//
// For each image it generates a slice of embeddings, one per layers of the model (93 in total), starting
// from the embeddings closest to the image, all the way to the last embedding closest to
// the output of the model.
func InceptionV3PerLayerEmbeddings(ctx *context.Context, images []*Node) (embeddings [][]*Node) {
	must.M(inceptionv3.DownloadAndUnpackWeights(InceptionV3Dir))
	ctxInception := ctx.In("inceptionv3").Checked(false)
	g := images[0].Graph()
	embeddings = make([][]*Node, len(images))
	for imgIdx, img := range images {
		g.PushAliasScope(fmt.Sprintf("img_%d", imgIdx))
		// Add batch dimension = 1.
		img = ExpandAxes(img, 0)
		_ = inceptionv3.BuildGraph(ctxInception, img).
			WithAliases(true).
			Trainable(false).
			PreTrained(InceptionV3Dir).
			Done()
		embeddings[imgIdx] = make([]*Node, InceptionV3NumLayers)
		for layerNum := range InceptionV3NumLayers {
			layerOutput := g.GetNodeByAlias(fmt.Sprintf("inceptionV3/conv_%03d/output", layerNum))
			if layerOutput == nil {
				exceptions.Panicf("couldn't file layer #%d for InceptionV3", layerNum)
			}
			embeddings[imgIdx][layerNum] = layerOutput
		}
		g.PopAliasScope()
	}
	return
}
