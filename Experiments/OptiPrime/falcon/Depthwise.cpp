// Standalone ICCAD 2023 Falcon packing reproduction for public test vectors.
#include <LinearLayer/ConvOptiEncode.h>
#include <LinearOperator/Conversion.h>
#include <Utils/prg.h>
#include <fstream>
#include <iostream>
#include <random>

#include "Layer.h"
namespace {
using FalconExperiment::Shape;
using Datatype::Tensor;
using HE::unified::UnifiedCiphertext;
using HE::HEEvaluator;
constexpr size_t t = 1 << 20;
size_t Up(size_t a, size_t b) { return (a+b-1)/b; }

void Run(HEEvaluator &he, const Shape &s, size_t id) {
    const size_t side=Up(s.h,s.stride);
    Tensor<uint64_t> x({s.channels,s.h,s.h}), clear(x.shape()), w({s.channels,1,3,3}),
                     bias({s.channels}), expected({s.channels,side,side});
    std::mt19937_64 rng(121+id);
    for (size_t i=0; i<x.size(); ++i) {
        clear(i)=rng()%256;
        auto share=rng()&(t-1);
        x(i)=he.server ? share : (clear(i)-share)&(t-1);
    }
    for (size_t i=0; i<w.size(); ++i) w(i)=(rng()%16-8)&(t-1);
    for (size_t c=0; c<s.channels; ++c) for (size_t y=0; y<side; ++y) for (size_t z=0; z<side; ++z) {
        uint64_t sum=0;
        for (size_t r=0;r<3;++r) for (size_t col=0;col<3;++col) {
            size_t iy=y*s.stride+r-1, ix=z*s.stride+col-1;
            if (iy<s.h && ix<s.h) sum+=clear((c*s.h+iy)*s.h+ix)*w(c*9+r*3+col);
        }
        expected((c*side+y)*side+z)=sum&(t-1);
    }
    if (!he.server) std::fill(w.data().begin(),w.data().end(),0);
    for (int ours=0;ours<=1;++ours) {
        LinearLayer::OptiEncodeOptions opt;
        opt.depthwise=true; opt.compact_output=true; opt.ciphertext_inverse_normalization=true;
        opt.threads=8; opt.add_bias=false;
        std::unique_ptr<LinearLayer::Conv2DOptiEncode> layer;
        if (ours) layer=std::make_unique<LinearLayer::Conv2DOptiEncode>(s.h,s.stride,1,w,bias,&he,opt);
        size_t nin=0,nout=0,mult=0,automorphisms=0;
        std::unique_ptr<FalconExperiment::Layer> falcon;
        if (!ours) falcon=std::make_unique<FalconExperiment::Layer>(he,s,w,opt.threads);
        auto before=he.IO->counter;
        auto out=ours ? (*layer)(x) : falcon->Forward(x);
        auto bytes=he.IO->counter-before;
        if (ours) {
            const auto &l=layer->Layout();
            nin=l.InputCiphertexts(); nout=l.OutputCiphertexts();
            mult=l.channels*nout; automorphisms=(l.baby_steps+l.giant_steps-2)*nout;
        } else { nin=falcon->InputCount(); nout=falcon->OutputCount(); mult=nout; }
        if (he.server) {
            Tensor<uint64_t> other(out.shape());
            he.IO->recv_data(other.data().data(),other.size()*8);
            for (size_t i=0;i<out.size();++i)
                if (((out(i)+other(i))&(t-1))!=expected(i))
                    throw std::runtime_error("Reference mismatch at layer "+std::to_string(id)+" index "+std::to_string(i));
        } else he.IO->send_data(out.data().data(),out.size()*8);
        std::cout<<"ROW "<<id<<" "<<(ours?"opti":"falcon")<<" "<<s.h<<" "<<s.channels<<" "<<s.stride
                 <<" "<<nin<<" "<<nout<<" "<<mult<<" "<<automorphisms<<" "<<bytes<<" "<<out.size()<<std::endl;
    }
}
}

int main(int argc,char **argv) {
    try {
        if (argc!=5) throw std::invalid_argument("Usage: executable party port degree plan");
        int party=std::stoi(argv[1]);
        Utils::NetIO io(party==1 ? nullptr : "127.0.0.1",std::stoi(argv[2]));
        size_t degree=std::stoul(argv[3]);
        HEEvaluator he(&io,party,degree,20,Datatype::HOST,{50,50},t);
        he.GenerateNewKey(LinearLayer::Conv2DOptiEncode::GaloisElements(degree));
        std::ifstream in(argv[4]);
        Shape s{}; size_t count=0;
        while(in>>s.h>>s.channels>>s.stride>>s.rows>>s.cols>>s.cx>>s.cw) Run(he,s,++count);
        if(!in.eof() || !count) throw std::runtime_error("Invalid plan");
        he.FreeKey();
    } catch(const std::exception &e) { std::cerr<<e.what()<<std::endl; return 1; }
}
