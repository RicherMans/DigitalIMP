'''
Created on Nov 19, 2014

@author: richman
'''
import argparse
from scipy import misc
import numpy as np

def main():
    args = parseArgs()
    interpolator = methods[args.method]
    if args.rotate:
        ox, oy = args.inputimage.shape
        for angle in args.rotate:
            rotatedimg = interpolator.rotate_image(args.inputimage, angle * np.pi / 180, ox / 2, oy / 2)
            misc.imsave('rotatedimg_' + str(angle) + '_' + args.method + '.tif', rotatedimg)
    if args.scale:
        for scale in args.scale:
            scaledimg = interpolator.scale_image(args.inputimage, scale)
            misc.imsave('scaledimg_' + str(scale) + '_' + args.method + '.tif', scaledimg)
    if args.translate:
        for translate in args.translate:
            dx,dy = translate
            translatedimg = interpolator.translate_image(args.inputimage,dx,dy)
            misc.imsave('tranlatedimg_'+str(dx)+'_'+str(dy)+'_'+args.method+'.tif', translatedimg)
        
def rotate(x, y, theta, ox, oy):
    """Rotate arrays of coordinates x and y by theta radians about the
    point (ox, oy).
    """
    s, c = np.sin(theta), np.cos(theta)
    x, y = np.asarray(x) - ox, np.asarray(y) - oy
    return x * c - y * s + ox, x * s + y * c + oy


class Bilinear():
    
    def translate_image(self, src, ox, oy):
        bx,by = src.shape
        dw,dh = bx+ox, oy+by
        dx, dy = np.meshgrid(np.arange(dw), np.arange(dh))
        return self._interpolate(src, dx, dy)
        
#         
    
    def rotate_image(self, src, theta, ox, oy):
        # Images have origin at the top left, so negate the angle.
        theta = -theta
        # Dimensions of source image. Note that scipy.misc.imread loads
        # images in row-major order, so src.shape gives (height, width).
        sh, sw = src.shape
        # Rotated positions of the corners of the source image.
        cx, cy = rotate([0, sw, sw, 0], [0, 0, sh, sh], theta, ox, oy)
        # Determine dimensions of destination image.
        dw, dh = (int(np.ceil(c.max() - c.min())) for c in (cx, cy))
        # Coordinates of pixels in destination image.
        dx, dy = np.meshgrid(np.arange(dw), np.arange(dh))
        # Corresponding coordinates in source image. Since we are
        # transforming dest-to-src here, the rotation is negated.
        sx, sy = rotate(dx + cx.min(), dy + cy.min(), -theta, ox, oy)
    #     sx, sy = rotate(dx + cx.min(), dy + cy.min(), -theta, ox, oy)
        # Select nearest neighbour/bilinear
        return self._interpolate(src, sx, sy)
        
    def _interpolate(self, im, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
    
        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1
#         Clip the x0 and x1, so that we dont have to check later if they are in range of the img
        x0 = np.clip(x0, 0, im.shape[1] - 1);
        x1 = np.clip(x1, 0, im.shape[1] - 1);
        y0 = np.clip(y0, 0, im.shape[0] - 1);
        y1 = np.clip(y1, 0, im.shape[0] - 1);
        Ia = im[ y0, x0 ]
        Ib = im[ y1, x0 ]
        Ic = im[ y0, x1 ]
        Id = im[ y1, x1 ]
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)
        
        return (wa * Ia + wb * Ib + wc * Ic + wd * Id)
    
    def scale_image(self, src, scale):
        ow, oh = src.shape
        dw, dh = int(ow * scale), int(oh * scale)
        
        x_ratio = ow / float(dw)
        y_ratio = oh / float(dh)
        
        px, py = np.meshgrid(np.arange(dw), np.arange(dh))
        # Shrinkage
        return self._interpolate(src, px * x_ratio, py * y_ratio)
    
class NearestNeighbor():
    
    def translate_image(self, src, ox, oy):
        bx,by = src.shape
        dw,dh = int(bx+ox), int(oy+by)
        dx, dy = np.meshgrid(np.arange(dw), np.arange(dh))
        
        sx, sy = self._interpolate(dx, dy)
        mask = (0 <= sx) & (bx > sx) & (0 <= sy) & (by > sy)
        dest = np.empty(shape=(dh, dw), dtype=src.dtype)
        # Copy valid coordinates from source image.
        dest[dy[mask], dx[mask]] = src[sy[mask], sx[mask]]
        return dest
    
    def scale_image(self, src, scale):
        ow, oh = src.shape
        dw, dh = int(ow * scale), int(oh * scale)
        
        x_ratio = ow / float(dw)
        y_ratio = oh / float(dh)
        
        px, py = np.meshgrid(np.arange(dw), np.arange(dh))
        # Shrinkage
        if x_ratio >= 1:
            sx, sy = self._interpolate(px * x_ratio, py * y_ratio)
            mask = (0 <= sx) & (0 < dw) & (0 <= sy) & (0 < dh)
        # Upscaling
        else:
            sx, sy = self._interpolate(px * x_ratio, py * y_ratio)
            mask = (0 <= sx) & (sx < dw) & (0 <= sy) & (sy < dh)
        # Create destination image.
        dest = np.empty(shape=(dw, dh), dtype=src.dtype)
        # Copy valid coordinates from source image.
        dest[py[mask], px[mask]] = src[sy[mask], sx[mask]]
        return dest
                
    def rotate_image(self, src, theta, ox, oy):
        """Rotate the image src by theta radians about (ox, oy).
        Pixels in the result that don't correspond to pixels in src are
        replaced by the value fill.
        """
        # Images have origin at the top left, so negate the angle.
        theta = -theta
    
        # Dimensions of source image. Note that scipy.misc.imread loads
        # images in row-major order, so src.shape gives (height, width).
        sh, sw = src.shape
    
        # Rotated positions of the corners of the source image.
        cx, cy = rotate([0, sw, sw, 0], [0, 0, sh, sh], theta, ox, oy)
    
        # Determine dimensions of destination image.
        dw, dh = (int(np.ceil(c.max() - c.min())) for c in (cx, cy))
    
        # Coordinates of pixels in destination image.
        dx, dy = np.meshgrid(np.arange(dw), np.arange(dh))
        # Corresponding coordinates in source image. Since we are
        # transforming dest-to-src here, the rotation is negated.
        sx, sy = rotate(dx + cx.min(), dy + cy.min(), -theta, ox, oy)
    #     sx, sy = rotate(dx + cx.min(), dy + cy.min(), -theta, ox, oy)
        # Select nearest neighbour/bilinear
        sx, sy = self._interpolate(sx, sy)
    #     sx, sy = sx.round().astype(int), sy.round().astype(int)
        # Mask for valid coordinates.
        mask = (0 <= sx) & (sx < sw) & (0 <= sy) & (sy < sh)
        # Create destination image.
        dest = np.empty(shape=(dh, dw), dtype=src.dtype)
        # Copy valid coordinates from source image.
        dest[dy[mask], dx[mask]] = src[sy[mask], sx[mask]]
        return dest

    def _interpolate(self, sx, sy):
        return np.floor(sx).astype(int), np.floor(sy).astype(int)

methods = {'nn':NearestNeighbor(), 'bi':Bilinear()}

def translatetype(strin):
    return [float(x) for x in strin.split(',')] 
    
def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputimage', type=misc.imread)
    parser.add_argument('-r', '--rotate', type=float, help='rotates the image at the specified angle, multiple angles also possible', nargs='+')
    parser.add_argument('-t', '--translate', type=translatetype, help='Translates the image,by a given x,y', nargs='+')
    parser.add_argument('-s', '--scale', type=float, help='Scales the image by a certain factor', nargs='+')
    parser.add_argument('-m', '--method', help='Methods, either bilinear or nearest neighbor', choices=methods, default=methods.keys()[0])
    parser.add_argument('-o', type=str, help='Outputfile')
    return parser.parse_args()

if __name__ == '__main__':
    main()
