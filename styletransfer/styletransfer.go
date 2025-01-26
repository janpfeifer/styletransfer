// Package styletransfer implements helper functions for doing StyleTransfer.
//
// It supports:
//
//   - "A Neural Algorithm of Artistic Style" 2015 Gatys, Ecker & Bethge [https://arxiv.org/abs/1508.06576]
//     implementation. See StyleTransfer.
//     See also article "Neural Style Transfer (NST) -- theory and implementation",
//     https://medium.com/@ferlatti.aldo/neural-style-transfer-nst-theory-and-implementation-c26728cf969d
//   - UI: DisplayImages on a Jupyter notebook using github.com/janpfeifer/gonb/gonbui
//   - I/O: LoadScaledImages
package styletransfer

import (
	"fmt"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/tensors"
	"time"
)

const (
	// ParamNumSteps is the hyperparameter that defines the number of steps to execute for transfer.
	// Defaults to 1000.
	ParamNumSteps = "num_steps"

	// ParamOriginalWeight is the weight on which to match the original image. Also known as the "alpha" parameter.
	ParamOriginalWeight = "original_weight"

	// ParamStyleWeight is the weight on which to match the style image. Also known as the "beta" parameter.
	ParamStyleWeight = "style_weight"
)

// Config for style transfer. Create is with New, and when finished configuring execute Config.Transfer to
// run the style transfer.
type Config struct {
	backend                             backends.Backend
	ctx                                 *context.Context
	original, style                     *tensors.Tensor
	originalLossWeight, styleLossWeight float64
	numSteps                            int
	numMomentsForStyle                  int
}

// New creates a style transfer configuration object: it takes as input the original image,
// the style image as tensors (color values from 0 to 1) and a context ctx with hyperparameters
// and used to create the style transfer model and the generated image.
//
// You can further configure the style transfer, and when done, call Config.Transfer to get execute the style transfer
// and get the generated image back.
//
// The context given can be saved (checkpoints), and later loaded in case you want to run more steps on the image.
//
// It uses the original style transfer algorithm described in [1] and [2].
//
// [1] "A Neural Algorithm of Artistic Style", 2015, Gatys, Ecker & Bethge -- https://arxiv.org/abs/1508.06576
// [2] "Neural Style Transfer (NST) -- theory and implementation", 2021 -- https://medium.com/@ferlatti.aldo/neural-style-transfer-nst-theory-and-implementation-c26728cf969d
func New(backend backends.Backend, ctx *context.Context, original, style *tensors.Tensor) *Config {
	normalizeExec := NewExec(backend, func(img *Node) *Node {
		// Images comes with values from 0.0 to 1.0, but inception was trained with values from -1.0 to 1.0.
		return MulScalar(AddScalar(MulScalar(img, 2.0), -1), 0.9)
	})
	cfg := &Config{
		backend:            backend,
		ctx:                ctx,
		original:           normalizeExec.Call(original)[0],
		style:              normalizeExec.Call(style)[0],
		originalLossWeight: context.GetParamOr(ctx, ParamOriginalWeight, 1.0),
		styleLossWeight:    context.GetParamOr(ctx, ParamStyleWeight, 1.0e4),
		numSteps:           context.GetParamOr(ctx, ParamNumSteps, 1000),
	}
	return cfg
}

// OriginalLossWeight sets the weight to use when matching the original image.
// In the original paper (see [1] and [2]) it is called "alpha".
func (cfg *Config) OriginalLossWeight(weight float64) *Config {
	cfg.originalLossWeight = weight
	return cfg
}

// StyleLossWeight sets the weight to use when matching the style image. Default to 0.5.
// In the original paper (see [1] and [2]) it is called "beta".
func (cfg *Config) StyleLossWeight(weight float64) *Config {
	cfg.styleLossWeight = weight
	return cfg
}

// NumSteps configures the number of steps to take during the style transfer.
//
// It defaults to the hyperparameter "num_steps", or if that is not set, defaults to 1000.
func (cfg *Config) NumSteps(numSteps int) *Config {
	cfg.numSteps = numSteps
	return cfg
}

// MomentsForStyle sets the number of moments to use from the style: if numMoments is > 0,
// it will use the referred moments of the distribution (1=mean, 2=mean+variance, 3=mean+variance+skew, etc.)
// for the style loss for transfer.
//
// The default is numMoments is 0, in which case it uses the GramMatrix for loss.
func (cfg *Config) MomentsForStyle(numMoments int) *Config {
	cfg.numMomentsForStyle = numMoments
	return cfg
}

// gramMatrix returns a [numChannels, numChannels] matrix with the correlation of channels across the image.
func gramMatrix(img *Node) *Node {
	numChannels := img.Shape().Dim(-1)
	flat := Reshape(img, -1, numChannels)
	gram := MatMul(Transpose(flat, 0, 1), flat)
	gram.AssertDims(numChannels, numChannels)
	return gram
}

// precalculateEmbeddings for the original image and store them as variables in the context.
func (cfg *Config) precalculateEmbeddings() {
	// Executes the GoMLX code to update the variables. It doesnt' return anything directly (only through
	// the updated variables.
	_ = context.ExecOnceN(cfg.backend, cfg.ctx, func(ctx *context.Context, original, style *Node) []*Node {
		g := original.Graph()
		ctx.SetTraining(g, false)
		allLayers := InceptionV3PerLayerEmbeddings(ctx, []*Node{original, style})
		layersNames := []string{"original_embeddings", "style_embeddings"}
		for imageIdx, layers := range allLayers {
			scopedCtx := ctx.In(layersNames[imageIdx])
			for layerIdx, layer := range layers {
				varName := fmt.Sprintf("layer_%d", layerIdx)
				// Create variable, or set it, if it doesn't yet exist.
				v := scopedCtx.GetVariable(varName)
				if v == nil {
					v = scopedCtx.VariableWithValueGraph(varName, layer)
				} else {
					v.SetValueGraph(layer)
				}
				v.SetTrainable(false)
			}
		}
		return nil
	}, cfg.original, cfg.style)
}

// loadEmbeddings load the values created by precalculatedEmbeddings in the scopedCtx --
// scopedCtx must be in scope "original_embeddings" or "style_embeddings".
func (cfg *Config) loadEmbeddings(scopedCtx *context.Context, g *Graph) []*Node {
	var e []*Node
	layerIdx := 0
	for {
		varName := fmt.Sprintf("layer_%d", layerIdx)
		v := scopedCtx.GetVariable(varName)
		if v == nil {
			// No more layers
			break
		}
		e = append(e, v.ValueGraph(g))
		layerIdx++
	}
	return e
}

// loss calculates the loss of x with respect to the upper layers original image and lower layers of the style image.
func (cfg *Config) loss(ctx *context.Context, x *Node) (loss *Node) {
	// Calculate the embeddings for all layers.
	g := x.Graph()
	xLayers := InceptionV3PerLayerEmbeddings(ctx, []*Node{x})[0]
	originalLayers := cfg.loadEmbeddings(ctx.In("original_embeddings"), g)
	styleLayers := cfg.loadEmbeddings(ctx.In("style_embeddings"), g)
	if len(xLayers) != len(originalLayers) {
		exceptions.Panicf("expected same number of layers for x (%d) and original (%d) layers",
			len(xLayers), len(originalLayers))
	}
	if len(xLayers) != len(styleLayers) {
		exceptions.Panicf("expected same number of layers for x (%d) and style (%d) layers",
			len(xLayers), len(styleLayers))
	}

	// Calculate mean loss on the style image.
	var styleLoss, originalLoss *Node
	for layerIdx := range InceptionV3NumLayers {
		xLayer := xLayers[layerIdx]
		originalLayer := originalLayers[layerIdx]
		styleLayer := styleLayers[layerIdx]

		numChannels := xLayer.Shape().Dim(-1)
		imageSize := xLayer.Shape().Dim(1) * xLayer.Shape().Dim(2)

		// Style loss: Gram Matrix loss
		gramX := gramMatrix(xLayer)
		gramStyle := gramMatrix(styleLayer)
		gramLoss := ReduceAllMean(Square(Sub(gramX, gramStyle)))
		gramLoss = DivScalar(gramLoss, 4*imageSize*numChannels)
		if styleLoss == nil {
			styleLoss = gramLoss
		} else {
			styleLoss = Add(styleLoss, gramLoss)
		}

		// Original image loss: mean of the square of the difference of the features between generated and new image.
		l2Loss := ReduceAllMean(Square(Sub(xLayer, originalLayer)))
		if originalLoss == nil {
			originalLoss = l2Loss
		} else {
			originalLoss = Add(originalLoss, l2Loss)
		}
	}

	// Normalize by the number of layers included in the loss.
	styleLoss = DivScalar(styleLoss, InceptionV3NumLayers)
	originalLoss = DivScalar(originalLoss, InceptionV3NumLayers)

	// Weight the loss terms and return:
	loss = Add(
		MulScalar(originalLoss, cfg.originalLossWeight),
		MulScalar(styleLoss, cfg.styleLossWeight))
	return
}

// transferStepGraph builds the computation graph that executes one step of style transfer.
func (cfg *Config) transferStepGraph(ctx *context.Context, g *Graph, xVar *context.Variable) *Node {
	ctx.SetTraining(g, true)
	x := xVar.ValueGraph(g)
	loss := cfg.loss(ctx, x)

	// Optimize loss on xVar
	opt := optimizers.FromContext(ctx)
	opt.UpdateGraph(ctx, g, loss)

	// Clip values of the generated image x to be back at -1 to 1.0 range.
	// If we don't do this, the images gets full of "specks" of dust, on pixels that "overflow" the value.
	x = xVar.ValueGraph(g) // Value has been updated by the optimizer, we need to fetch it again.
	x = ClipScalar(x, -1, 1)
	xVar.SetValueGraph(x)
	return loss
}

// Transfer style and returns the newly generated new image.
// It can be called multiple times, each time continues with the generated image from
// where the previous one left of.
func (cfg *Config) Transfer() *tensors.Tensor {
	ctx := cfg.ctx

	// Pre-generate embeddings for original and style images.
	cfg.precalculateEmbeddings()

	// Target image x that we want to generate: make it a trainable variable, initialized with the original image.
	xVar := ctx.GetVariable("x")
	if xVar == nil {
		// Create target image from original one.
		xT := tensors.FromShape(cfg.original.Shape())
		xT.CopyFrom(cfg.original)
		xVar = ctx.VariableWithValue("x", xT)
	}
	xVar.SetTrainable(true)

	// Create computation graph for one training step.
	// It updates x and returns the loss -- only for displaying.
	stepExec := context.NewExec(cfg.backend, ctx, func(ctx *context.Context, g *Graph) *Node {
		return cfg.transferStepGraph(ctx, g, xVar)
	})

	// Iterate training step, minimizing the loss and generating the image with the style transferred.
	var avgDuration float64
	var lastPrint time.Time
	var loss *tensors.Tensor
	for step := 0; step < cfg.numSteps; step++ {
		start := time.Now()
		if loss != nil {
			loss.FinalizeAll()
		}
		loss = stepExec.Call()[0]
		duration := time.Since(start).Seconds()
		if step < 10 {
			avgDuration = duration
		} else {
			avgDuration = 0.9*avgDuration + 0.1*duration
		}
		if time.Since(lastPrint) > time.Second {
			fmt.Printf("\rStyle transferring: step=%05d of %05d (%8.1f ms/step) -- loss=%s			       ",
				step+1, cfg.numSteps, avgDuration*1000.0, loss)
			lastPrint = time.Now()
		}
	}
	fmt.Printf("\rStyle transferring: step=%05d of %05d (%5.1fms/step) -- loss=%s			  \n",
		cfg.numSteps, cfg.numSteps, avgDuration*1000.0, loss)

	// Take generated image variable and de-normalize it back to a normal image.
	x := xVar.Value()
	x = ExecOnce(cfg.backend, func(img *Node) *Node {
		return DivScalar(AddScalar(img, 1), 2)
	}, x)
	return x
}
