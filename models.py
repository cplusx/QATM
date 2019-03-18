from __future__ import print_function, division
import keras
import tensorflow as tf
import keras.backend as K
class QATM( keras.layers.Layer ):
    def __init__( self, alpha=10, **kwargs ):
        self.alpha = alpha
        super( QATM, self ).__init__( **kwargs )
  
    def build( self, input_shape ):
        self.coef_ref = self.add_weight(shape=(1,),
                                    initializer=keras.initializers.Constant(self.alpha),
                                    name='softmax_coef_ref',)
        self.coef_qry = self.add_weight(shape=(1,),
                                    initializer=keras.initializers.Constant(self.alpha),
                                    name='softmax_coef_qry',)
        
        super( QATM, self ).build( input_shape )
        
    def call( self, x ):
        batch_size, ref_row, ref_col, qry_row, qry_col = [ tf.shape(x)[k] for k in range(5) ]
        x = tf.reshape( x, [batch_size, ref_row * ref_col, qry_row * qry_col ] )
        xm_ref = x - K.max(x,axis=1,keepdims=True)
        conf_ref = tf.nn.softmax( self.coef_ref*xm_ref, axis=1 )
        xm_qry = x - K.max(x,axis=2,keepdims=True)
        conf_qry = tf.nn.softmax( self.coef_qry*xm_qry, axis=2 )
        confidence = K.sqrt(conf_ref * conf_qry )
        conf_values, ind3 = tf.nn.top_k( confidence, k=1 ) # batch_size, ref_size, 1
        ind1, ind2 = tf.meshgrid( tf.range( batch_size ), 
                                  tf.range( ref_row * ref_col ), indexing='ij' )
        ind1 = K.flatten( ind1 )
        ind2 = K.flatten( ind2 )
        ind3 = K.flatten( ind3 )
        indices = K.stack([ind1,ind2,ind3],axis=1)
        values = tf.gather_nd( confidence, indices )
        values = tf.reshape( values, [batch_size, ref_row, ref_col, 1])
        return values
    
    def compute_output_shape( self, input_shape ):
        bs, H, W, _, _ = input_shape
        return (bs, H, W, 1)
class MyNormLayer( keras.layers.Layer ):
    def __init__( self, **kwargs ):
        super( MyNormLayer, self ).__init__( **kwargs )
    
    def build( self, x ):
        super( MyNormLayer, self ).build(x)
        
    def call( self, x ):
        x1, x2 = x
        bs, H, W, _ = [tf.shape(x1)[i] for i in range(4)]
        _, h, w, _ = [tf.shape(x2)[i] for i in range(4)]
        x1 = tf.reshape(x1, ( bs, H*W, -1 ) )
        x2 = tf.reshape(x2, ( bs, h*w, -1 ) )
        concat = tf.concat([x1, x2], axis=1)
        x_mean = K.mean( concat, axis=1, keepdims=True )
        x_std = K.std( concat, axis=1, keepdims = True )
        x1 = (x1 - x_mean) / x_std
        x2 = (x2 - x_mean) / x_std
        x1 = tf.reshape(x1, ( bs, H, W, -1 ) )
        x2 = tf.reshape(x2, ( bs, h, w, -1 ) )
        return [x1, x2]