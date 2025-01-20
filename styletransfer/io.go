package styletransfer

import (
	"bytes"
	"fmt"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/tensors/images"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/janpfeifer/gonb/gonbui"
	"github.com/janpfeifer/must"
	"github.com/pkg/errors"
	_ "golang.org/x/image/webp"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"
)

// DisplayImages using gonbui.
// It only works in a notebook.
func DisplayImages(imgs ...*tensors.Tensor) {
	buf := &bytes.Buffer{}
	fmt.Fprintf(buf, "<table><tr>\n")
	for _, img := range imgs {
		src := must.M1(gonbui.EmbedImageAsPNGSrc(images.ToImage().Single(img)))
		fmt.Fprintf(buf, "  <td><img src=\"%s\"/></td>\n", src)
	}
	fmt.Fprintf(buf, "</tr></table>\n")
	gonbui.DisplayHTMLF(buf.String())
}

// LoadImage as a tensor shaped [height, width, 3], with the image normalized for inceptionv3 (values from -1.0 to 1.0)
//
// Image type is taken from its extension, .png and .jpg are accepted.
func LoadImage(imagePath string) (imgT *tensors.Tensor, err error) {
	imgFile, err := os.Open(imagePath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to open image in %s", imagePath)
	}
	defer func() { _ = imgFile.Close() }()

	img, _, err := image.Decode(imgFile)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to decode image in %s", imagePath)
	}
	imgT = images.ToTensor(dtypes.Float32).Single(img)
	return
}

// LoadScaledImages loads the standard "original.png" and "style.png" images and scale
// it to InceptionV3 sizes, and normalizes them to values -1.0 to 1.0.
func LoadScaledImages(backend backends.Backend, originalPath, stylePath string) (original, style *tensors.Tensor) {
	original = must.M1(LoadImage(originalPath))
	style = must.M1(LoadImage(stylePath))
	fmt.Println("Images:")
	fmt.Printf("- original:\t%s\n", original.Shape())
	fmt.Printf("- style:   \t%s\n", style.Shape())
	original = InceptionV3ResizeTensor(backend, original)
	style = InceptionV3ResizeTensor(backend, style)
	fmt.Printf("\t> Scaled to %s\n", original.Shape())
	return
}
